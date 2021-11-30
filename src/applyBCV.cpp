#include <iostream>
#include <fstream>
#include <math.h>
#include <sys/time.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <string.h>
#include <map>
#include <sstream>
#include <x86intrin.h>
#include <pthread.h>
#include <thread>
#include "zlib.h"
#include <sys/stat.h>

#ifdef TORCH_EXTENSION_NAME
    #include <torch/extension.h>
#else
    #include <torch/script.h>
#endif

using namespace std;

//some global variables
int RAND_SAMPLES; //will all be set later (if needed)
int image_m; int image_n; int image_o; int image_d=1;
float SSD0=0.0; float SSD1=0.0; float SSD2=0.0; float distfx_global; float beta=1;
//float SIGMA=8.0;
int qc=1;

//struct for multi-threading of mind-calculation
struct mind_data{
	float* im1;
    float* d1;
    uint64_t* mindq;
    int qs;
    int ind_d1;
};



struct parameters{
    float alpha; int levels=0; bool segment,affine,rigid;
    vector<int> grid_spacing; vector<int> search_radius;
    vector<int> quantisation;
    string fixed_file,moving_file,output_stem,moving_seg_file,affine_file,deformed_file;
};

#include "imageIOgzType.h"
#include "transformations.h"
//#include "primsMST.h"
//#include "regularisation.h"
//#include "MINDSSCbox.h"
#include "dataCostD.h"
#include "parseArguments.h"


int main (int argc, char * const argv[]) {
	//Initialise random variable


    //PARSE INPUT ARGUMENTS

    if(argc<4||argv[1][1]=='h'){
        cout<<"=============================================================\n";
        cout<<"Usage (required input arguments):\n";
        cout<<"./applyBCV -M moving.nii.gz -O output -D deformed.nii.gz \n";
        cout<<"optional parameters:\n";
        cout<<" -A <affine_matrix.txt> \n";
        cout<<"=============================================================\n";
        return 1;
    }

    parameters args;
    parseCommandLine(args, argc, argv);

    size_t split_def=args.deformed_file.find_last_of("/\\");
    if(split_def==string::npos){
        split_def=-1;
    }
    size_t split_moving=args.moving_file.find_last_of("/\\");
    if(split_moving==string::npos){
        split_moving=-1;
    }


    if(args.deformed_file.substr(args.deformed_file.length()-2)!="gz"){
        cout<<"images must have nii.gz format\n";
        return -1;
    }
    if(args.moving_file.substr(args.moving_file.length()-2)!="gz"){
        cout<<"images must have nii.gz format\n";
        return -1;
    }

    cout<<"Transforming "<<args.moving_file.substr(split_moving+1)<<" into "<<args.deformed_file.substr(split_def+1)<<"\n";




    //READ IMAGES and INITIALISE ARRAYS

    timeval time1,time2,time1a,time2a;


    short* seg2;
    int M,N,O,P; //image dimensions

    //==ALWAYS ALLOCATE MEMORY FOR HEADER ===/
    char* header=new char[352];
    //TODO: Read nifti
    readNifti(args.moving_file,seg2,header,M,N,O,P);

    image_m=M; image_n=N; image_o=O;

    int m=image_m; int n=image_n; int o=image_o;
    int sz=m*n*o; //sz == total image size over all dimensions



    //READ AFFINE MATRIX from linearBCV if provided (else start from identity)

    float* X=new float[16];

    if(args.affine){
        size_t split_affine=args.affine_file.find_last_of("/\\");
        if(split_affine==string::npos){
            split_affine=-1;
        }

        //TODO read mat.txt
        cout<<"Reading affine matrix file: "<<args.affine_file.substr(split_affine+1)<<"\n";
        ifstream matfile;
        matfile.open(args.affine_file);
        for(int i=0;i<4;i++){
            string line;
            getline(matfile,line);
            sscanf(line.c_str(),"%f  %f  %f  %f",&X[i],&X[i+4],&X[i+8],&X[i+12]);
        }
        matfile.close();


    }
    else{
        cout<<"Using identity transform.\n";
        fill(X,X+16,0.0f);
        X[0]=1.0f; X[1+4]=1.0f; X[2+8]=1.0f; X[3+12]=1.0f;
    }

    for(int i=0;i<4;i++){
        printf("%+4.3f | %+4.3f | %+4.3f | %+4.3f \n",X[i],X[i+4],X[i+8],X[i+12]);//X[i],X[i+4],X[i+8],X[i+12]);

    }




    string inputflow;
    inputflow.append(args.output_stem);
    inputflow.append("_displacements.dat");

    cout<<"Reading displacements from:\n"<<inputflow<<"\n";


    cout<<"=============================================================\n";

    vector<float> flow=readFile<float>(inputflow);

    int sz3=flow.size()/3;
    int grid_step=round(pow((float)sz/(float)sz3,0.3333333));
    // reduce grid to sqrt_3(3*full_size/flow_size) (uniform grid step in every direction)

    cout<<"grid step "<<grid_step<<"\n";

    int step1; int hw1; float quant1;

    //set initial flow-fields to 0; i indicates backward (inverse) transform
    //u is in x-direction (2nd dimension), v in y-direction (1st dim) and w in z-direction (3rd dim)

    // init 3dim flow xyz <-> uvw * LEN*WIDTH*HEIGHT -> full size flow field
    float* ux=new float[sz]; float* vx=new float[sz]; float* wx=new float[sz];
    for(int i=0;i<sz;i++){
        ux[i]=0.0;
        vx[i]=0.0;
        wx[i]=0.0;
    }

    int m1,n1,o1,sz1;
    m1=m/grid_step; n1=n/grid_step; o1=o/grid_step; sz1=m1*n1*o1;
    float* u1=new float[sz1]; float* v1=new float[sz1]; float* w1=new float[sz1];

    //Fill gridded flow field vectors. orig flow field ordering= [x0,x1,x2, at y0,z0; x0,x1,x2 at y1,z0 ...]
    for(int i=0;i<sz1;i++){
        u1[i]=flow[i];
        v1[i]=flow[i+sz1];
        w1[i]=flow[i+sz1*2];

    }
    //TODO Upsample, in: full size flow field (returned value), reduced flow field, full dimension image size, reduced flow field size
    upsampleDeformationsCL(ux,vx,wx,u1,v1,w1,m,n,o,m1,n1,o1);



    short* segw=new short[sz];
    //TODO Fill
    fill(segw,segw+sz,(short)0);
    //TODO warpAffine
    //output segw = seg_warped
    warpAffineS(segw,seg2,X,ux,vx,wx);


    // TODO Write nifti (p=1 -> 3d image if o > 1), copy header
    gzWriteSegment(args.deformed_file,segw,header,m,n,o,1);




	return 0;
}

int64_t applyBCV_main(int64_t _argc, std::vector<std::string> _argv) {
    std::vector<const char *> argv;
    argv.reserve(_argv.size() + 1);
    for(auto it = std::begin(_argv); it != std::end(_argv); ++it) {
        argv.push_back(it->c_str());
    }
    argv.push_back(nullptr);  // needed to terminate the args list

    return main(_argc, const_cast<char* const *>(argv.data()));
}

torch::Tensor applyBCV_jacobian(
    torch::Tensor input_u,
    torch::Tensor input_v,
    torch::Tensor input_w,
    torch::Tensor input_factor) {

    float* u = input_u.data_ptr<float>();
    float* v = input_v.data_ptr<float>();
    float* w = input_w.data_ptr<float>();
    int* factor = input_factor.data_ptr<int>();

    int m = input_u.size(2);
    int n = input_u.size(1);
    int o = input_u.size(0);

    // cout<<"m"<<m;
    // cout<<"n"<<n;
    // cout<<"o"<<o;

    float jacobian_output = jacobian(u, v, w, m, n, o, *factor);
    std::vector<float> jac_vect{jacobian_output};

    auto options = torch::TensorOptions();
    return torch::from_blob(jac_vect.data(), {1}, options).clone();
}


torch::Tensor applyBCV_interp3(
    torch::Tensor pInput,
    torch::Tensor pX1,
    torch::Tensor pY1,
    torch::Tensor pZ1,
    torch::Tensor pOutput_size,
    torch::Tensor pFlag) {

    int m2 = pInput.size(0);
    int n2 = pInput.size(1);
    int o2 = pInput.size(2);

    int m = pOutput_size[0].item<int>();
    int n = pOutput_size[1].item<int>();
    int o = pOutput_size[2].item<int>();

    float* input = pInput.data_ptr<float>();
    float* interp=new float[m*n*o];

    float* x1 = pX1.data_ptr<float>();
    float* y1 = pY1.data_ptr<float>();
    float* z1 = pZ1.data_ptr<float>();

    bool* flag = pFlag.data_ptr<bool>();

    interp3(
        interp, // interpolated output
	    input, // gridded flow field
		x1, y1, z1, //helper var (output size)
		m, n, o, //output size
		m2, n2, o2, //gridded flow field size
		*flag
    );

    std::vector<float> interp_vect{interp, interp + m*n*o};

    auto options = torch::TensorOptions();
    return torch::from_blob(interp_vect.data(), {m,n,o}, options).clone();
}

torch::Tensor applyBCV_volfilter(
    torch::Tensor pInput,
    torch::Tensor pKernel_sz,
    torch::Tensor pSigma) {

    int m = pInput.size(0);
    int n = pInput.size(1);
    int o = pInput.size(2);

    int Kernel_sz = pKernel_sz.item<int>();
    float Sigma = pSigma.item<float>();

    float* input = pInput.data_ptr<float>();

    volfilter(input, m, n, o, Kernel_sz, Sigma);

    std::vector<float> gauss_vect{input, input + m*n*o};

    auto options = torch::TensorOptions();
    return torch::from_blob(gauss_vect.data(), {m,n,o}, options).clone();
}

std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor > applyBCV_consistentMappingCL(

    torch::Tensor pInput_u,
    torch::Tensor pInput_v,
    torch::Tensor pInput_w,
    torch::Tensor pInput_u2,
    torch::Tensor pInput_v2,
    torch::Tensor pInput_w2,
    torch::Tensor input_factor) {

    int m = pInput_u.size(0);
    int n = pInput_u.size(1);
    int o = pInput_u.size(2);

    torch::Tensor input_u_copy = pInput_u.clone();
    torch::Tensor input_v_copy = pInput_v.clone();
    torch::Tensor input_w_copy = pInput_w.clone();
    torch::Tensor input_u2_copy = pInput_u2.clone();
    torch::Tensor input_v2_copy = pInput_v2.clone();
    torch::Tensor input_w2_copy = pInput_w2.clone();

    float* u = input_u_copy.data_ptr<float>();
    float* v = input_v_copy.data_ptr<float>();
    float* w = input_w_copy.data_ptr<float>();
    float* u2 = input_u2_copy.data_ptr<float>();
    float* v2 = input_v2_copy.data_ptr<float>();
    float* w2 = input_w2_copy.data_ptr<float>();

    int* factor = input_factor.data_ptr<int>();

    // cout<<"m"<<m;
    // cout<<"n"<<n;
    // cout<<"o"<<o;

    consistentMappingCL(u, v, w, u2, v2, w2, m, n, o, *factor);

    std::vector<float> new_u{u, u + m*n*o};
    std::vector<float> new_v{v, v + m*n*o};
    std::vector<float> new_w{w, w + m*n*o};
    std::vector<float> new_u2{u2, u2 + m*n*o};
    std::vector<float> new_v2{v2, v2 + m*n*o};
    std::vector<float> new_w2{w2, w2 + m*n*o};

    auto options = torch::TensorOptions();

    return std::tuple<
            torch::Tensor,
            torch::Tensor,
            torch::Tensor,
            torch::Tensor,
            torch::Tensor,
            torch::Tensor>(
        torch::from_blob(new_u.data(), {m,n,o}, options).clone(),
        torch::from_blob(new_v.data(), {m,n,o}, options).clone(),
        torch::from_blob(new_w.data(), {m,n,o}, options).clone(),
        torch::from_blob(new_u2.data(), {m,n,o}, options).clone(),
        torch::from_blob(new_v2.data(), {m,n,o}, options).clone(),
        torch::from_blob(new_w2.data(), {m,n,o}, options).clone()
    );
}


std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor> applyBCV_upsampleDeformationsCL(

    torch::Tensor pInput_u,
    torch::Tensor pInput_v,
    torch::Tensor pInput_w,
    torch::Tensor pInput_u2,
    torch::Tensor pInput_v2,
    torch::Tensor pInput_w2) {

    int m = pInput_u.size(0);
    int n = pInput_u.size(1);
    int o = pInput_u.size(2);

    int m2 = pInput_u2.size(0);
    int n2 = pInput_u2.size(1);
    int o2 = pInput_u2.size(2);

    torch::Tensor input_u_copy = pInput_u.clone();
    torch::Tensor input_v_copy = pInput_v.clone();
    torch::Tensor input_w_copy = pInput_w.clone();
    torch::Tensor input_u2_copy = pInput_u2.clone();
    torch::Tensor input_v2_copy = pInput_v2.clone();
    torch::Tensor input_w2_copy = pInput_w2.clone();

    float* u = input_u_copy.data_ptr<float>();
    float* v = input_v_copy.data_ptr<float>();
    float* w = input_w_copy.data_ptr<float>();
    float* u2 = input_u2_copy.data_ptr<float>();
    float* v2 = input_v2_copy.data_ptr<float>();
    float* w2 = input_w2_copy.data_ptr<float>();


    // cout<<"m"<<m;
    // cout<<"n"<<n;
    // cout<<"o"<<o;

    upsampleDeformationsCL(u, v, w, u2, v2, w2, m, n, o, m2, n2, o2);

    std::vector<float> new_u{u, u + m*n*o};
    std::vector<float> new_v{v, v + m*n*o};
    std::vector<float> new_w{w, w + m*n*o};

    auto options = torch::TensorOptions();

    return std::tuple<
            torch::Tensor,
            torch::Tensor,
            torch::Tensor>(
        torch::from_blob(new_u.data(), {m,n,o}, options).clone(),
        torch::from_blob(new_v.data(), {m,n,o}, options).clone(),
        torch::from_blob(new_w.data(), {m,n,o}, options).clone()
    );
}

torch::Tensor applyBCV_warpAffineS(
    torch::Tensor image_in,
    torch::Tensor pInput_T,
    torch::Tensor pInput_u1,
    torch::Tensor pInput_v1,
    torch::Tensor pInput_w1) {

    
    torch::Tensor input_image_copy = image_in.clone();
    torch::Tensor input_u1_copy = pInput_u1.clone();
    torch::Tensor input_v1_copy = pInput_v1.clone();
    torch::Tensor input_w1_copy = pInput_w1.clone();
    torch::Tensor input_T_copy = pInput_T.clone();
    

    int m = pInput_u1.size(0);
    int n = pInput_u1.size(1);
    int o = pInput_u1.size(2);
    float* u1 = input_u1_copy.data_ptr<float>();
    float* v1 = input_v1_copy.data_ptr<float>();
    float* w1 = input_w1_copy.data_ptr<float>();
    float* T = input_T_copy.data_ptr<float>();

    short* input_img = input_image_copy.data_ptr<short>();
    short* warp= new short[m*n*o];

    warpAffineS(warp,input_img,T,u1,v1,w1);

    std::vector<short> warp_vect{warp, warp+ m*n*o};

    auto options = torch::TensorOptions();
    return torch::from_blob(warp_vect.data(), {m,n,o}, options).clone();
}




#ifdef TORCH_EXTENSION_NAME
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("applyBCV_main", &applyBCV_main, "applyBCV_main");
        m.def("applyBCV_jacobian", &applyBCV_jacobian, "applyBCV_jacobian");
        m.def("applyBCV_interp3", &applyBCV_interp3, "applyBCV_interp3");
        m.def("applyBCV_volfilter", &applyBCV_volfilter,"applyBCV_volfilter");
        m.def("applyBCV_consistentMappingCL", &applyBCV_consistentMappingCL,"applyBCV_consistentMappingCL");
        m.def("applyBCV_upsampleDeformationsCL", &applyBCV_upsampleDeformationsCL,"applyBCV_upsampleDeformationsCL");
        m.def("applyBCV_warpAffineS", &applyBCV_warpAffineS,"applyBCV_warpAffineS");
    }

#else
    TORCH_LIBRARY(deeds_applyBCV, m) {
        m.def("applyBCV_main", &applyBCV_main);
        m.def("applyBCV_jacobian", &applyBCV_jacobian);
        m.def("applyBCV_interp3", &applyBCV_interp3);
        m.def("applyBCV_volfilter", &applyBCV_volfilter);
        m.def("applyBCV_consistentMappingCL", &applyBCV_consistentMappingCL);
        m.def("applyBCV_upsampleDeformationsCL", &applyBCV_upsampleDeformationsCL);
        m.def("applyBCV_warpAffineS", &applyBCV_warpAffineS);

    }
#endif
