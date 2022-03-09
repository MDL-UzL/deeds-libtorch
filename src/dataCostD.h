
void interp3xyz(float* datai,float* data,float* datax,float* datay,int len1,int len2){
    //x-interp
    for(int k=0;k<len1;k++){
        for(int j=0;j<len2;j++){
            int j2=(j+1)/2;
            if(j%2==1){
                for(int i=0;i<len1;i++){
                    datax[i+j*len1+k*len1*len2]=data[i+j2*len1+k*len1*len1];
                }
            }
            else
            for(int i=0;i<len1;i++){
                datax[i+j*len1+k*len1*len2]=0.5*(data[i+j2*len1+k*len1*len1]+data[i+(j2+1)*len1+k*len1*len1]);
            }
        }
    }


    //y-interp
    for(int k=0;k<len1;k++){
        for(int j=0;j<len2;j++){
            for(int i=0;i<len2;i++){
                int i2=(i+1)/2;
                if(i%2==1)
                datay[i+j*len2+k*len2*len2]=datax[i2+j*len1+k*len1*len2];
                else
                datay[i+j*len2+k*len2*len2]=0.5*(datax[i2+j*len1+k*len1*len2]+datax[i2+1+j*len1+k*len1*len2]);
            }
        }
    }

    //z-interp
    for(int k=0;k<len2;k++){
        int k2=(k+1)/2;
        if(k%2==1){
            for(int j=0;j<len2;j++){
                for(int i=0;i<len2;i++){
                    datai[i+j*len2+k*len2*len2]=datay[i+j*len2+k2*len2*len2];
                }
            }

        }
        else{
            for(int j=0;j<len2;j++){
                for(int i=0;i<len2;i++){
                    datai[i+j*len2+k*len2*len2]=0.5*(datay[i+j*len2+k2*len2*len2]+datay[i+j*len2+(k2+1)*len2*len2]);
                }
            }
        }
    }

}

void interp3xyzB(float* datai,float* data,float* datax,float* datay,int len1,int len2){
    //x-interp
    for(int k=0;k<len1;k++){
        for(int j=0;j<len2;j++){
            int j2=(j+1)/2;
            if(j%2==0){
                for(int i=0;i<len1;i++){
                    datax[i+j*len1+k*len1*len2]=data[i+j2*len1+k*len1*len1];
                }
            }
            else
            for(int i=0;i<len1;i++){
                datax[i+j*len1+k*len1*len2]=0.5*(data[i+j2*len1+k*len1*len1]+data[i+(j2-1)*len1+k*len1*len1]);
            }
        }
    }


    //y-interp
    for(int k=0;k<len1;k++){
        for(int j=0;j<len2;j++){
            for(int i=0;i<len2;i++){
                int i2=(i+1)/2;
                if(i%2==0)
                datay[i+j*len2+k*len2*len2]=datax[i2+j*len1+k*len1*len2];
                else
                datay[i+j*len2+k*len2*len2]=0.5*(datax[i2+j*len1+k*len1*len2]+datax[i2-1+j*len1+k*len1*len2]);
            }
        }
    }

    //z-interp
    for(int k=0;k<len2;k++){
        int k2=(k+1)/2;
        if(k%2==0){
            for(int j=0;j<len2;j++){
                for(int i=0;i<len2;i++){
                    datai[i+j*len2+k*len2*len2]=datay[i+j*len2+k2*len2*len2];
                }
            }

        }
        else{
            for(int j=0;j<len2;j++){
                for(int i=0;i<len2;i++){
                    datai[i+j*len2+k*len2*len2]=0.5*(datay[i+j*len2+k2*len2*len2]+datay[i+j*len2+(k2-1)*len2*len2]);
                }
            }
        }
    }

}


void dataCostCL(
    unsigned long* data,
    unsigned long* data2,
    float* results,
    int m,int n,int o,int len2,int step1,int hw,float quant,float alpha,int randnum){
    // Returns mind sums of patches (voxel size images are reduced to patches) given two mind-images

    int len=hw*2+1;
    len2=pow(hw*2+1,3); //len2 is calculated again (see linearBCV.cpp)
    //hw is search radius

    int sz=m*n*o; //full size
    int m1=m/step1; int n1=n/step1; int o1=o/step1; //gridded cell size
    int sz1=m1*n1*o1;

    //cout<<"len2: "<<len2<<" sz1= "<<sz1<<"\n";



    int quant2=quant;

    //const int hw2=hw*quant2; == pad1
    // quant means dilation, hw means search window width == search steps
    int pad1=quant2*hw; //pad stop = quant * search_radius
    int pad2=pad1*2; // uniform padding =  2*  quant * search_radius

    int mp=m+pad2;//m_padded - uniform padding for all x,y,z
    int np=n+pad2;//n_padded
    int op=o+pad2;//o_padded
    int szp=mp*np*op;
    unsigned long* data2p=new unsigned long[szp];

    // std::cout<<"\ndata=";
    // for(int pri=0;pri<m*n*o;pri++){
    //     std::cout<<data[pri]<<" ";
    // }
    // std::cout<<"\n";
    //     std::cout<<"\ndata2=";
    // for(int pri=0;pri<m*n*o;pri++){
    //     std::cout<<data2[pri]<<" ";
    // }
    // std::cout<<"\n";
    // std::cout<<"\ndata2p=";
    // for(int pri=0;pri<szp;pri++){
    //     std::cout<<data2p[pri]<<" ";
    // }
    std::cout<<"\n";


    // apply padding quant * search_radius in all 8 directions
    for(int k=0;k<op;k++){
        for(int j=0;j<np;j++){
            for(int i=0;i<mp;i++){
                data2p[i+j*mp+k*mp*np]=data2[max(min(i-pad1,m-1),0) // 0 to x_max_idx but with x offset (shift of half padding to center data)
                                      +max(min(j-pad1,n-1),0)*m
                                      +max(min(k-pad1,o-1),0)*m*n]; //replication padding
            }
        }
    }


    int skipz=1; int skipx=1; int skipy=1; //define kernel steps to be skipped
    if(step1>4){
        if(randnum>0){
            //true for linearBCV and deedsBCV
            skipz=2; skipx=2;
        }
        if(randnum>1){
            //wrong for linearBCV
            skipy=2;
        }
    }
    if(randnum>1&step1>7){
        //wrong for linearBCV
        skipz=3; skipx=3; skipy=3;
    }
    if(step1==4&randnum>1)
    //wrong for linearBCV
    skipz=2;


    float maxsamp=ceil((float)step1/(float)skipx)
                 *ceil((float)step1/(float)skipz)
                 *ceil((float)step1/(float)skipy); //maxsamples per kernel microsteps in x,y,z (sparse)
    //printf("randnum: %d, maxsamp: %d ",randnum,(int)maxsamp);


    float alphai=(float)step1/(alpha*(float)quant); //linearBCV alpha=1

    float alpha1=0.5*alphai/(float)(maxsamp); //alpha1 = step/(alpha*quant) // this scales the mind datacost against the quadratic offset distance regularisation

    //unsigned long buffer[1000];

#pragma omp parallel for
    // iterate over patches
    for(int z=0;z<o1;z++){ // iterate gridded z
        for(int x=0;x<n1;x++){ // iterate gridded y
            for(int y=0;y<m1;y++){ //iterate gridded x
                int z1=z*step1;
                int x1=x*step1;
                int y1=y*step1; //for every patch coordinate x,y,z get corner starting voxel coordinate of the patch, position of patch1
                /*for(int k=0;k<step1;k++){
                    for(int j=0;j<step1;j++){
                        for(int i=0;i<step1;i++){
                            buffer[i+j*step1+k*step1*step1]=data[i+y1+(j+x1)*m+(k+z1)*m*n];
                        }
                    }
                }*/

                for(int l=0;l<len2;l++){ //iterate over kernel_entries
                    int out1=0; //will be summed up
                    int zs=l/(len*len);
                    int xs=(l-zs*len*len)/len;
                    int ys=l-zs*len*len-xs*len;
                    //get position of search (ys increases faster than xs than zs)
                    // (xs,ys,zs) is center of search kernel

                    zs*=quant;
                    xs*=quant;
                    ys*=quant; // apply dilation to search coordinates

                    int x2=xs+x1;
                    int z2=zs+z1;
                    int y2=ys+y1; // per (x1,z1,y1) we add relative search position (xs, zs, ys) // position of patch2
                    for(int k=0;k<step1;k+=skipz){ //iterate steps, skip one, skip two etc. i.e. sparse microsteps
                        for(int j=0;j<step1;j+=skipx){
                            for(int i=0;i<step1;i+=skipy){
                                //iterate over every voxel of a patch
                                //unsigned int t=buffer[i+j*STEP+k*STEP*STEP]^buf2p[i+j*mp+k*mp*np];
                                //out1+=(wordbits[t&0xFFFF]+wordbits[t>>16]);
                                unsigned long t1=data   [i+y1 + (j+x1)*m+   (k+z1)*m*n];//buffer[i+j*step1+k*step1*step1];
                                unsigned long t2=data2p [i+y2 + (j+x2)*mp+  (k+z2)*mp*np];//last term is search offset
                                // t2=data2p            [i+y2   (j+x2)*mp+  (k+z2)*mp*np]; // y2,x2,z2 contain search offset
                                std::cout<<"\npopcountll: "<<__builtin_popcountll(t1^t2)<<"\n";
                                out1+=__builtin_popcountll(t1^t2); //bitwise xor -> are mind features either both1 or both zero? count ones in bitstream // patch2 - patch1
                                //count differences in mind features per voxel in a patch for every search position
                                //patch_cube_size * search_cube_size dimension
                                //out one is patch cost.
                            }
                        }
                    }
                    results[(y+x*m1+z*m1*n1)*len2+l]=out1*alpha1; // divide by samplecount of microsteps
                    // store scaled patch costs as difference of patches (center_patch - search patch around search position) -> What would be the cost if the center patch is moved to the search position?
                    // alpha1 also scales this cost agains the quadratic offset model build up in regularisationCL. There the cost is just added to the datacost valued calculated here
                }
            }
        }
    }


    delete[] data2p;

    return;


}


void warpImageCL(float* warped,float* im1,float* im1b,float* u1,float* v1,float* w1){
    int m=image_m;
    int n=image_n;
    int o=image_o;
    int sz=m*n*o;

    float ssd=0;
    float ssd0=0;
    float ssd2=0;

    interp3(warped,im1,u1,v1,w1,m,n,o,m,n,o,true);

    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            for(int k=0;k<o;k++){
                ssd+=pow(im1b[i+j*m+k*m*n]-warped[i+j*m+k*m*n],2);
                ssd0+=pow(im1b[i+j*m+k*m*n]-im1[i+j*m+k*m*n],2);
            }
        }
    }

    ssd/=m*n*o;
    ssd0/=m*n*o;
    SSD0=ssd0;
    SSD1=ssd;

}

void warpAffineS(short* warped,short* input,float* X,
                 float* u1,float* v1,float* w1, int m=image_m, int n=image_n, int o=image_o){ // flow field
    // int m=image_m;
    // int n=image_n;
    // int o=image_o;
    //     std::cout<<"\nshort input_img=";
    // for(int pri=0;pri<m*n*o ;pri++){
	// 	std::cout<<input[pri]<<" ";
	// }
    //         std::cout<<"\nfloat X=";
    // for(int pri=0;pri<12 ;pri++){
	// 	std::cout<<X[pri]<<" ";
	// }
    //             std::cout<<"\nfloat u1=";
    // for(int pri=0;pri<m*n*o ;pri++){
	// 	std::cout<<u1[pri]<<" ";
	// }
    int sz=m*n*o;
    for(int k=0;k<o;k++){
        for(int j=0;j<n;j++){
            for(int i=0;i<m;i++){
                // affine transformation (every x + y + z) + flow field displacement -> get x,y,z for lookup in input image
                float y1=(float)i*X[0]+(float)j*X[1]+(float)k*X[2]+(float)X[3]+v1[i+j*m+k*m*n];
                float x1=(float)i*X[4]+(float)j*X[5]+(float)k*X[6]+(float)X[7]+u1[i+j*m+k*m*n];
                float z1=(float)i*X[8]+(float)j*X[9]+(float)k*X[10]+(float)X[11]+w1[i+j*m+k*m*n];

                //do not interpolate looked up values
                int x=round(x1); int y=round(y1);  int z=round(z1);

                //if(y>=0&x>=0&z>=0&y<m&x<n&z<o){
                    //lookup x,y,z and store to warped image
                    warped[i+j*m+k*m*n]=input[min(max(y,0),m-1)+min(max(x,0),n-1)*m+min(max(z,0),o-1)*m*n];
                //}
                //else{
                //    warped[i+j*m+k*m*n]=0;
                //}
            }
        }
    }

    // no distance metric here
                // std::cout<<"\nshort warped=";
    // for(int pri=0;pri<m*n*o ;pri++){
	// 	std::cout<<warped[pri]<<" ";
	// }
}
void warpAffine(float* warped,float* input,float* im1b,float* X,float* u1,float* v1,float* w1, int m=image_m, int n=image_n, int o=image_o){
    int sz=m*n*o;

    float ssd=0;
    float ssd0=0;
    float ssd2=0;

    for(int k=0;k<o;k++){
        for(int j=0;j<n;j++){
            for(int i=0;i<m;i++){

                float y1=(float)i*X[0]+(float)j*X[1]+(float)k*X[2]+(float)X[3]+v1[i+j*m+k*m*n];
                float x1=(float)i*X[4]+(float)j*X[5]+(float)k*X[6]+(float)X[7]+u1[i+j*m+k*m*n];
                float z1=(float)i*X[8]+(float)j*X[9]+(float)k*X[10]+(float)X[11]+w1[i+j*m+k*m*n];

                //interpolate looked up values (see also interp3)
                int x=floor(x1); int y=floor(y1);  int z=floor(z1);
                float dx=x1-x; float dy=y1-y; float dz=z1-z;


                warped[i+j*m+k*m*n]=(1.0-dx)*(1.0-dy)*(1.0-dz)*input[min(max(y,0),m-1)
                +min(max(x,0),n-1)*m+min(max(z,0),o-1)*m*n]
                +(1.0-dx)*dy*(1.0-dz) * input[min(max(y+1,0),m-1)+min(max(x,0),n-1)*m+min(max(z,0),o-1)*m*n]
                +dx*(1.0-dy)*(1.0-dz) * input[min(max(y,0),m-1)+min(max(x+1,0),n-1)*m+min(max(z,0),o-1)*m*n]
                +(1.0-dx)*(1.0-dy)*dz * input[min(max(y,0),m-1)+min(max(x,0),n-1)*m+min(max(z+1,0),o-1)*m*n]
                +dx*dy*(1.0-dz) *       input[min(max(y+1,0),m-1)+min(max(x+1,0),n-1)*m+min(max(z,0),o-1)*m*n]
                +(1.0-dx)*dy*dz *       input[min(max(y+1,0),m-1)+min(max(x,0),n-1)*m+min(max(z+1,0),o-1)*m*n]
                +dx*(1.0-dy)*dz *       input[min(max(y,0),m-1)+min(max(x+1,0),n-1)*m+min(max(z+1,0),o-1)*m*n]
                +dx*dy*dz *             input[min(max(y+1,0),m-1)+min(max(x+1,0),n-1)*m+min(max(z+1,0),o-1)*m*n];
            }
        }
    }

    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            for(int k=0;k<o;k++){
                //squared(intensity differences)
                ssd+=pow(im1b[i+j*m+k*m*n]-warped[i+j*m+k*m*n],2);
                ssd0+=pow(im1b[i+j*m+k*m*n]-input[i+j*m+k*m*n],2);
            }
        }
    }

    ssd/=m*n*o; // normalize with total size
    ssd0/=m*n*o;
    SSD0=ssd0;
    SSD1=ssd;


}

// libtorch unittest API
torch::Tensor datacost_d_warpAffineS(
    torch::Tensor moving,
    torch::Tensor pInput_T,
    torch::Tensor pInput_u1,
    torch::Tensor pInput_v1,
    torch::Tensor pInput_w1) {

    torch::Tensor moving_copy = moving.clone();
    torch::Tensor input_u1_copy = pInput_u1.clone();
    torch::Tensor input_v1_copy = pInput_v1.clone();
    torch::Tensor input_w1_copy = pInput_w1.clone();
    torch::Tensor input_T_copy = pInput_T.clone();

    int m = moving.size(0);
    int n = moving.size(1);
    int o = moving.size(2);
    float* u1 = input_u1_copy.data_ptr<float>();
    float* v1 = input_v1_copy.data_ptr<float>();
    float* w1 = input_w1_copy.data_ptr<float>();
    float* T = input_T_copy.data_ptr<float>();

    short* input_moving = moving_copy.data_ptr<short>();
    short* warped= new short[m*n*o];

    warpAffineS(warped,input_moving,T,u1,v1,w1, m, n, o);
    //             std::cout<<"\nshort warp=";
    // for(int pri=0;pri<m*n*o ;pri++){
	// 	std::cout<<warp[pri]<<" ";
	// }
    std::vector<short> warp_vect{warped, warped+m*n*o};

    auto options = torch::TensorOptions().dtype(torch::kShort);
    return torch::from_blob(warp_vect.data(), {m,n,o}, options).clone();
}

torch::Tensor datacost_d_warpAffine(
    torch::Tensor moving,
    torch::Tensor pInput_T,
    torch::Tensor pInput_u1,
    torch::Tensor pInput_v1,
    torch::Tensor pInput_w1) {

    torch::Tensor moving_copy = moving.clone();
    torch::Tensor input_u1_copy = pInput_u1.clone();
    torch::Tensor input_v1_copy = pInput_v1.clone();
    torch::Tensor input_w1_copy = pInput_w1.clone();
    torch::Tensor input_T_copy = pInput_T.clone();

    int m = moving.size(0);
    int n = moving.size(1);
    int o = moving.size(2);
    float* u1 = input_u1_copy.data_ptr<float>();
    float* v1 = input_v1_copy.data_ptr<float>();
    float* w1 = input_w1_copy.data_ptr<float>();
    float* T = input_T_copy.data_ptr<float>();

    float* input_moving = moving_copy.data_ptr<float>();
    float* dummy_compare_image = new float[m*n*o];
    float* warped = new float[m*n*o];

    warpAffine(warped,input_moving,dummy_compare_image,T,u1,v1,w1, m, n, o);
    std::vector<float> warp_vect{warped, warped+m*n*o};

    auto options = torch::TensorOptions().dtype(torch::kFloat);
    return torch::from_blob(warp_vect.data(), {m,n,o}, options).clone();
}

torch::Tensor datacost_d_datacostCL(torch::Tensor pMind_img_a, torch::Tensor pMind_img_b, torch::Tensor pGrid_divisor, torch::Tensor pHw, torch::Tensor pDilation, torch::Tensor pAlpha) {

	// Prepare input variables
    torch::Tensor mind_img_a_copy = pMind_img_a.clone();
    torch::Tensor mind_img_b_copy = pMind_img_b.clone();

	int m = mind_img_a_copy.size(0);
    int n = mind_img_a_copy.size(1);
    int o = mind_img_a_copy.size(2);

    int64_t* mind_img_a = mind_img_a_copy.data_ptr<int64_t>();
    // std::cout<<"\nmind_img_a=";
    // for(int pri=0;pri<m*n*o;pri++){
    //     std::cout<<mind_img_a[pri]<<" ";
    // }
    // std::cout<<"\n";
    int64_t* mind_img_b = mind_img_b_copy.data_ptr<int64_t>();
    // unsigned long* mind_img_a_ulong = (unsigned long*)((void*)&mind_img_a);
    unsigned long* mind_img_a_ulong = reinterpret_cast<unsigned long*>(mind_img_a);
	unsigned long* mind_img_b_ulong = reinterpret_cast<unsigned long*>(mind_img_b);
    int grid_divisor = *(pGrid_divisor.data_ptr<int>());
    int hw = *(pHw.data_ptr<int>());
    int dilation = *(pDilation.data_ptr<int>());
    float alpha = *(pAlpha.data_ptr<float>());

	// Prepare output variables
    int results_len = m*n*o*pow(hw*2+1,3);
	float* results = new float[results_len];
    //len2 is 0 since its unused in fuction and RANDNUM = 1 since this is fixed for all calls in deeds
    dataCostCL(mind_img_a_ulong, mind_img_b_ulong, results, m, n, o, 0, grid_divisor, hw, dilation, alpha, 1);

	// Prepare lib output
	std::vector<float> results_vect{results, results+results_len};

    auto float_options = torch::TensorOptions().dtype(torch::kFloat);
	return torch::from_blob(results_vect.data(), {m/grid_divisor,n/grid_divisor,o/grid_divisor,int(pow(hw*2+1,3))}, float_options).clone();
}