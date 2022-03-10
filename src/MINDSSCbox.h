
void boxfilter(float* input,float* temp1,float* temp2,
    int hw, // mind_step
    int m,int n,int o){

    int sz=m*n*o;
    for(int i=0;i<sz;i++){
        temp1[i]=input[i];//copy to temp, initialization
    }
    ////y
    for(int k=0;k<o;k++){
        for(int j=0;j<n;j++){
            for(int i=1;i<m;i++){
                temp1[i+j*m+k*m*n]+=temp1[(i-1)+j*m+k*m*n]; //add intensity value but index starting delayed
            }
        }
    }

    for(int k=0;k<o;k++){ //iterate over all z dimensions
        for(int j=0;j<n;j++){ // for x dim do (or y dim?)
            for(int i=0;i<(hw+1);i++){ // [0 to hw]
                temp2[i+j*m+k*m*n]=temp1[(i+hw)+j*m+k*m*n]; // -0 at beginning, left of sliding index, copy offset y value?
            }
            for(int i=(hw+1);i<(m-hw);i++){ //[hw+1 to len-hw) ('box' can move free without hitting a border)
                temp2[i+j*m+k*m*n]=temp1[(i+hw)+j*m+k*m*n]-temp1[(i-hw-1)+j*m+k*m*n]; //delta (symmetric y val offset + dy / -dy)
            }
            for(int i=(m-hw);i<m;i++){ // [len-hw to len)
                temp2[i+j*m+k*m*n]=temp1[(m-1)+j*m+k*m*n]-temp1[(i-hw-1)+j*m+k*m*n]; //delta (last value of dim (fixed) - negative offseted value
            }
        }
    }
    ////x
    for(int k=0;k<o;k++){
        for(int j=1;j<n;j++){
            for(int i=0;i<m;i++){
                temp2[i+j*m+k*m*n]+=temp2[i+(j-1)*m+k*m*n]; //add intensity value but reversed axis (other)
            }
        }
    }

    for(int k=0;k<o;k++){
        for(int i=0;i<m;i++){
            for(int j=0;j<(hw+1);j++){ //caution, dimensions were switched
                temp1[i+j*m+k*m*n]=temp2[i+(j+hw)*m+k*m*n]; //see above, but for next dim (x)
            }
            for(int j=(hw+1);j<(n-hw);j++){
                temp1[i+j*m+k*m*n]=temp2[i+(j+hw)*m+k*m*n]-temp2[i+(j-hw-1)*m+k*m*n];
            }
            for(int j=(n-hw);j<n;j++){
                temp1[i+j*m+k*m*n]=temp2[i+(n-1)*m+k*m*n]-temp2[i+(j-hw-1)*m+k*m*n];
            }
        }
    }
    ////z
    //add intensity value but last reversed axis z
    for(int k=1;k<o;k++){
        for(int j=0;j<n;j++){
            for(int i=0;i<m;i++){
                temp1[i+j*m+k*m*n]+=temp1[i+j*m+(k-1)*m*n];
            }
        }
    }

    // see above, but now for last axis
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){
            for(int k=0;k<(hw+1);k++){
                input[i+j*m+k*m*n]=temp1[i+j*m+(k+hw)*m*n];
            }
            for(int k=(hw+1);k<(o-hw);k++){
                input[i+j*m+k*m*n]=temp1[i+j*m+(k+hw)*m*n]-temp1[i+j*m+(k-hw-1)*m*n];
            }
            for(int k=(o-hw);k<o;k++){
                input[i+j*m+k*m*n]=temp1[i+j*m+(o-1)*m*n]-temp1[i+j*m+(k-hw-1)*m*n];
            }
        }
    }
}


void imshift(float* input,float* output,int dx,int dy,int dz,int m,int n,int o){
    // Get values of shifted patch. If coordinate is not within image dimensions return same intensity as original image (will result in zero distance)
    //shift image with preservation of original image values when shifted area is out of bounds
    for(int k=0;k<o;k++){ //z
        for(int j=0;j<n;j++){ //x
            for(int i=0;i<m;i++){ //y , iterate orig image size
                if(i+dy>=0 && i+dy<m &&
                   j+dx>=0 && j+dx<n &&
                   k+dz>=0 && k+dz<o) //this is something like the min(max) construct
                    output[i+j*m+k*m*n]=input[i+dy+(j+dx)*m+(k+dz)*m*n]; //lookup displaced values and return
                else
                    output[i+j*m+k*m*n]=input[i+j*m+k*m*n]; //lookup value w/o displacement (just copy), sth like 'inversed mirroring ad edged = normal image at egdes'
            }
        }
    }
}

/*void *distances(void *threadarg)
{
	struct mind_data *my_data;
	my_data = (struct mind_data *) threadarg;
    float* im1=my_data->im1;
    float* d1=my_data->d1;
    int qs=my_data->qs;
    int ind_d1=my_data->ind_d1;
    int m=image_m;
    int n=image_n;
    int o=image_o;*/

void distances(float* im1,float* d1,int m,int n,int o,int qs,int l){
    int sz1=m*n*o;
	float* w1=new float[sz1];
    int len1=6;//not needed

    float* temp1=new float[sz1]; //img size temp
    float* temp2=new float[sz1]; //img size temp
    // int dx[6]={+qs, +qs, -qs, 0,   +qs, +0 }; //redifinition
	// int dy[6]={+qs, -qs, 0,   -qs, 0,   +qs}; //redefinition
	// int dz[6]={0,   0,   +qs, +qs, +qs, +qs}; //redefiniton
    int dx[6]={+qs, +qs, +qs, +qs, 0,   0}; //redifinition
	int dy[6]={+qs, -qs, 0,   0,   +qs, +qs}; //redefinition
	int dz[6]={0,   0,   +qs, -qs, +qs, -qs}; //redefiniton
    //dx, dy, dz could be passed directly to this function from upper call to omit redefinitions
	// Offset patches in every 6-neighbourhood direction by quanstisation step (radius)
    imshift(im1,w1,dx[l],dy[l],dz[l],m,n,o);
    // std::cout<<"\nw1=";
    // for(int pri=0;pri<m*n*o ;pri++){
    //     std::cout<<w1[pri]<<" ";
    // }
    // std::cout<<"\nw2=";
    // for(int pri=0;pri<sz_pad ;pri++){
    //     std::cout<<w2[pri]<<" ";
    // }

    for(int i=0;i<sz1;i++){
        w1[i]=(w1[i]-im1[i])*(w1[i]-im1[i]); //(0-im[i])^2 = squared img dist from intensity val
    }

    // for(int k=0;k<o;k++){
    //     for(int j=0;j<m;j++){
    //         for(int i=0;i<n;i++){
    //             int w2_coord = i-dx[l] + (j-dy[l])*n + (k-dz[l])*m*n;
    //             int im_coord = i + j*n + k*m*n;
    //             w2[w2_coord] = (w2[w2_coord] - im1[im_coord]) * (w2[w2_coord] - im1[im_coord]); //(0-im[i])^2 = squared img dist from intensity val
    //         }
    //     }
    // }
    // std::cout<<"\nsquared_patch_distance=";
    // for(int pri=0;pri<m*n*o ;pri++){
    //     std::cout<<w1[pri]<<" ";
    // }
    // for(int pri=0;pri<sz_pad ;pri++){
    //     std::cout<<w2[pri]<<" ";
    // }
    //3 dim box filter = sth. like blur
    // boxfilter(w1,temp1,temp2,qs,m,n,o); //w1 is input and output
    for(int i=0;i<sz1;i++){
        d1[i+l*sz1]=w1[i];
    }

    delete[] temp1; delete[] temp2; delete[] w1;
}

//__builtin_popcountll(left[i]^right[i]); absolute hamming distances
void descriptor(uint64_t* mindq,float* im1,
    int m,int n,int o, //image dims
    int qs, float* output_mind_twelve=0, float* output_mind_bare=0){ //mind_step (chain values smaller than quantisation chain)
	timeval time1,time2;
    //MIND with self-similarity context

    //3^3 shift combinations (+qs,0,-qs) (x-dir, y-dir, z-dir) but only adjacent placed to origin (no diagonals) = 6 combinations

	// int sx[12]={-qs,+0, -qs,+0, +0, +qs,+0, +0, +0, -qs,+0, +0}; //is that MIND-SSC?
	// int sy[12]={+0, -qs,+0, +qs,+0, +0, +0, +qs,+0, +0, +0, -qs};
	// int sz[12]={+0, +0, +0, +0, -qs,+0, -qs,+0, -qs,+0, -qs,+0};

	// int sy[12]={ +0, -qs, -qs, +0,  +0,  +qs, +0,  +0,  +0,  -qs, +0,  +0}; //is that MIND-SSC?
	// int sx[12]={-qs, +0,  +0,  +qs, +0,  +0,  +0,  +qs, +0,  +0,  +0,  -qs};
	// int sz[12]={+0,  +0,  +0,  +0,  -qs, +0,  -qs, +0,  -qs, +0,  -qs, +0};

    int sy[12]={0,   -qs, -qs, 0,  -qs, 0,   -qs, 0,   0,   0,   0,   0 }; //is that MIND-SSC?
	int sx[12]={-qs, 0,   0,   qs, 0,   0,   0,   0,   -qs, 0,   -qs, 0 };
	int sz[12]={0,   0,   0,   0,  0,   -qs, 0,   +qs, 0,   -qs, 0,   qs};

	int index[12]={0,0,1,1,2,2,3,3,4,4,5,5};

	float sigma=0.75;//1.0;//0.75;//1.5;
	int rho=ceil(sigma*1.5)*2+1; // is unused!

	int len1=6; //len dx dy dz
	const int len2=12; //len sx sy sz

    image_d=12;
	int d=12;
    int sz1=m*n*o;

    pthread_t thread1, thread2, thread3;


    //============== DISTANCES USING BOXFILTER ===================
	float* d1=new float[sz1*len1]; //img_size * (dx,dy,dz) (6 neighbourhood)
    gettimeofday(&time1, NULL);

#pragma omp parallel for
    for(int l=0;l<len1;l++){ //for all dx, dy, dz
        //l iterator controls the shift package (dx,dy,dz)
        distances(im1,d1,m,n,o,qs,l); //d1 is returned (stored distances per x,y,z,l dimension, 4-dim)
    }
    std::cout<<"\ndistances=";
    for(int pri=0;pri<m*n*o*6 ;pri++){
        std::cout<<d1[pri]<<" ";
    }

    gettimeofday(&time2, NULL);
    float timeMIND1=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
    gettimeofday(&time1, NULL);

    //quantisation table
    const int val=6;

    const unsigned long long power=32;


#pragma omp parallel for
    // float mind_twelve[m*n*o*12];

    for(int k=0;k<o;k++){ //iterate z
        //could be moved out of loop
        unsigned int tablei[6]={0,1,3,7,15,31}; //lookup table
        float compare[val-1];
        for(int i=0;i<val-1;i++){
            compare[i]=-log((i+1.5f)/val);//compare is const for every iteration
        }
        //could be moved out of loop

        float mind1[12];//is this correct here? -> could be moved before l_iter loop (only needed within l_iter loop)

        for(int j=0;j<n;j++){//iterate x
            for(int i=0;i<m;i++){ //iterate y
                std::cout<<"\nmind1_bare: ";
                for(int l=0;l<len2;l++){ // iterate (sx,sy,sz) / index[l] package to get 12 mind values
                    int eff_sx = min(max(0, i+sx[l]),m-1);
                    int eff_sy = min(max(0, j+sy[l]),n-1);
                    int eff_sz = min(max(0, k+sz[l]),o-1);
                    // consecutive l's will take same d1 offset (0...6) but different spatial coordinate
                    mind1[l]=d1[eff_sx + eff_sy*m + eff_sz*m*n+index[l]*sz1]; //mind1 is 1-dim, size=12
                    // -> take same node but calc 2 diffs per node and add? 2*6diffs = 12
                    // if(i+sx[l]>=0 && i+sx[l]<m && j+sy[l]>=0 && j+sy[l]<n && k+sz[l]>=0 && k+sz[l]<o){ //min(max) construct
                    //     mind1[l]=d1[i+sx[l]+(j+sy[l])*m+(k+sz[l])*m*n+index[l]*sz1]; //mind1 is 1-dim, size=12
                    //     //read offseted distance value
                    // }
                    // else{
                    //     mind1[l]=d1[i+j*m+k*m*n+index[l]*sz1]; //(sz1=m*n*o) builds 12-neighbourhood
                    //     //read without (sx,sy,sz) ofset but with l=12 layer offset
                    // }
                }
                float minval=*min_element(mind1,mind1+len2); //get minimum value of all 12 stored mind1 features
                float sumnoise=0.0f;
                for(int l=0;l<len2;l++){
                    mind1[l]-=minval; //reset minimum value of mind1 to 0 and lower others accordingly
                    std::cout<<mind1[l]<<" ";
                    output_mind_bare[i+j*m+k*m*n+l*sz1] = mind1[l];
                    sumnoise+=mind1[l]; //accumulate
                }
                float noise1=max(sumnoise/(float)len2,1e-6f); //rescale accumulated mind features
                for(int l=0;l<len2;l++){
                    mind1[l]/=noise1;
                }
                unsigned long long accum=0;
                unsigned long long tabled1=1;

                for(int l=0;l<len2;l++){ //iterate over all mind features
                    //mind1[l]=exp(-mind1[l]);
                    int mind1val=0;
                    for(int c=0;c<val-1;c++){ //accumulate over 5 values
                        mind1val+=compare[c]>mind1[l]?1:0; //count if mind feature is smaller than compare const
                    }
                    //int mind1val=min(max((int)(mind1[l]*val-0.5f),0),val-1);
                    accum+=tablei[mind1val]*tabled1; //*32^l, propably this was done because of casting to long long
                    tabled1*=power;
                }
                mindq[i+j*m+k*m*n]=accum; //one mind value for every coordinate xyz
            }

        }
    }
    // std::cout<<"\nmindq=";
    // for(int pri=0;pri<m*n*o ;pri++){
	// 	std::cout<<mindq[pri]<<" ";
	// }
    // std::cout<<"\nmind_twelve=";
    // for(int pri=0;pri<m*n*o*12 ;pri++){
    //     std::cout<<output_mind_twelve[pri]<<" ";
    // }

    gettimeofday(&time2, NULL);
    float timeMIND2=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
    delete[] d1;


}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mind_ssc_descriptor(
    torch::Tensor image,
    torch::Tensor pQuantisation_step) {

    int o = image.size(0);
    int n = image.size(1);
    int m = image.size(2);

    torch::Tensor image_copy = image.clone();

    float* input_image = image_copy.data_ptr<float>();
    int* quantisation_step = pQuantisation_step.data_ptr<int>();

    uint64_t* output = new uint64_t[m*n*o];
    float* output_mind_twelve = new float[m*n*o*12];
    float* output_mind_bare = new float[m*n*o*12];

    descriptor(output, input_image, o, m, n, *quantisation_step, output_mind_twelve, output_mind_bare);
    std::vector<uint64_t> output_vect{output, output+m*n*o};
    std::vector<float> output_mind_bare_vect{output_mind_bare, output_mind_bare+m*n*o*12};
    std::vector<float> output_mind_twelve_vect{output_mind_twelve, output_mind_twelve+m*n*o*12};

    auto options = torch::TensorOptions().dtype(torch::kInt64);
    auto float_options = torch::TensorOptions().dtype(torch::kFloat);
    return std::tuple<
            torch::Tensor,
            torch::Tensor, torch::Tensor>(
        torch::from_blob(output_vect.data(), {n,m,o}, options).clone(),
        torch::from_blob(output_mind_twelve_vect.data(), {12,n,m,o}, float_options).clone(),
        torch::from_blob(output_mind_bare_vect.data(), {12,n,m,o}, float_options).clone()
    );
}
