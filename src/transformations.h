/* several functions to interpolate and symmetrise deformations
 calculates Jacobian and harmonic Energy */


void interp3(float* interp, // interpolated output
			 float* input, // gridded flow field
			 float* x1,float* y1,float* z1, //helper var (output size)
			 int m,int n,int o, //output size
			 int m2,int n2,int o2, //gridded flow field size
			 bool flag){

	// for(int pri=0;pri<m2*n2*o2 ;pri++){
	// 	std::cout<<input[pri]<<" ";
	// }

	// std::cout<<"\n";
	// for(int pri=0;pri<m*n*o ;pri++){
	// 	std::cout<<x1[pri]<<" ";
	// }
	for(int k=0;k<o;k++){ //iterate output z
		for(int j=0;j<n;j++){ // iterate output y
			for(int i=0;i<m;i++){ //iterate x
				int x=floor(x1[i+j*m+k*m*n]);
				int y=floor(y1[i+j*m+k*m*n]);
				int z=floor(z1[i+j*m+k*m*n]);
				float dx=x1[i+j*m+k*m*n]-x; float dy=y1[i+j*m+k*m*n]-y; float dz=z1[i+j*m+k*m*n]-z; // dx,dy,dz in gridded flow field relative coordinates

				if(flag){
					x+=j; y+=i; z+=k;
				}
				//trilinear interpolation: 8x partial cube volume from desired corner point * value of corner point
				interp[i+j*m+k*m*n]=
				//partial cube volume		//value at corner
											//x									//y											//z
				//clamp
				//clamping seems to be wrong (x <-> y clamping is mixed)

				//							//(0 ... x_interp ... x_idx max)	//(0 ... y_interp ... y_idx max)*x_idx_max	//(0 ... z_interp ... z_idx max)*x_idx_max*y_idx_max

				//reziprocal: when dx=1  	we want to have val at idx_x
				(1.0-dx)*(1.0-dy)*(1.0-dz)*	input[	min(max(y,0),m2-1)			+min(max(x,0),n2-1)*m2						+min(max(z,0),o2-1)*m2*n2]
				//reziprocal: when dx=0  	we want to have val at idx_x+1
				+dx*(1.0-dy)*(1.0-dz)*		input[	min(max(y,0),m2-1)			+min(max(x+1,0),n2-1)*m2					+min(max(z,0),o2-1)*m2*n2]
				+(1.0-dx)*dy*(1.0-dz)*		input[	min(max(y+1,0),m2-1)		+min(max(x,0),n2-1)*m2						+min(max(z,0),o2-1)*m2*n2]
				+(1.0-dx)*(1.0-dy)*dz*		input[	min(max(y,0),m2-1)			+min(max(x,0),n2-1)*m2						+min(max(z+1,0),o2-1)*m2*n2]

				+(1.0-dx)*dy*dz*			input[	min(max(y+1,0),m2-1)		+min(max(x,0),n2-1)*m2						+min(max(z+1,0),o2-1)*m2*n2]
				+dx*(1.0-dy)*dz*			input[	min(max(y,0),m2-1)			+min(max(x+1,0),n2-1)*m2					+min(max(z+1,0),o2-1)*m2*n2]
				+dx*dy*(1.0-dz)*			input[	min(max(y+1,0),m2-1)		+min(max(x+1,0),n2-1)*m2					+min(max(z,0),o2-1)*m2*n2]
											//3-dim indexing of flattened array
				+dx*dy*dz*					input[min(max(y+1,0),m2-1)			+min(max(x+1,0),n2-1)*m2					+min(max(z+1,0),o2-1)*m2*n2];
			}
		}
	}
	// for(int pri=0;pri<m*n*o ;pri++){
	// 	std::cout<<interp[pri]<<" ";
	// }
}



void filter1(float* imagein,float* imageout,int m,int n,int o,float* filter,
	int length,int dim){

	int i,j,k,f;
	int i1,j1,k1;
	int hw=(length-1)/2;

	for(i=0;i<(m*n*o);i++){
		imageout[i]=0.0;
	}

	for(k=0;k<o;k++){
		for(j=0;j<n;j++){
			for(i=0;i<m;i++){
				for(f=0;f<length;f++){
					//replicate-padding
					if(dim==1)
						imageout[i+j*m+k*m*n]+=filter[f]*imagein[max(min(i+f-hw,m-1),0)+j*m+k*m*n];
					if(dim==2)
						imageout[i+j*m+k*m*n]+=filter[f]*imagein[i+max(min(j+f-hw,n-1),0)*m+k*m*n];
					if(dim==3)
						imageout[i+j*m+k*m*n]+=filter[f]*imagein[i+j*m+max(min(k+f-hw,o-1),0)*m*n];
				}
			}
		}
	}
}

void volfilter(float* imagein,int m,int n,int o,int length,float sigma){
	// gaussian filter

	int hw=(length-1)/2;
	int i,j,f;
	float hsum=0;
	float* filter=new float[length];
	for(i=0;i<length;i++){
		filter[i]=exp(-pow((i-hw),2)/(2*pow(sigma,2)));
		hsum=hsum+filter[i];
	}
	for(i=0;i<length;i++){
		filter[i]=filter[i]/hsum;
	}
	float* image1=new float[m*n*o];
    for(i=0;i<m*n*o;i++){
        image1[i]=imagein[i];
    }
    filter1(image1,imagein,m,n,o,filter,length,1);
	filter1(imagein,image1,m,n,o,filter,length,2);
	filter1(image1,imagein,m,n,o,filter,length,3);

	delete[] image1;
	delete[] filter;

}



float jacobian(float* u1,float* v1,float* w1,int m,int n,int o,int factor){

	float factor1=1.0/(float)factor;
	float jmean=0.0;
	float jstd=0.0;
	int i;
	float grad[3]={-0.5,0.0,0.5};
	float* Jac=new float[m*n*o];

	float* J11=new float[m*n*o];
	float* J12=new float[m*n*o];
	float* J13=new float[m*n*o];
	float* J21=new float[m*n*o];
	float* J22=new float[m*n*o];
	float* J23=new float[m*n*o];
	float* J31=new float[m*n*o];
	float* J32=new float[m*n*o];
	float* J33=new float[m*n*o];

	for(i=0;i<(m*n*o);i++){
		J11[i]=0.0;
		J12[i]=0.0;
		J13[i]=0.0;
		J21[i]=0.0;
		J22[i]=0.0;
		J23[i]=0.0;
		J31[i]=0.0;
		J32[i]=0.0;
		J33[i]=0.0;
	}

	float neg=0; float Jmin=1; float Jmax=1; float J;
	float count=0; float frac;

	filter1(u1,J11,m,n,o,grad,3,2); // filter length always stays 3, direction of filter is changed 3 times per u1, v1, w1
	filter1(u1,J12,m,n,o,grad,3,1);
	filter1(u1,J13,m,n,o,grad,3,3);
	// cout<<"u1\n";
	// for (int i = 0; i < m*n*o; i++)
    // 	cout << u1[i] << " ";
	filter1(v1,J21,m,n,o,grad,3,2);
	filter1(v1,J22,m,n,o,grad,3,1);
	filter1(v1,J23,m,n,o,grad,3,3);

	filter1(w1,J31,m,n,o,grad,3,2);
	filter1(w1,J32,m,n,o,grad,3,1);
	filter1(w1,J33,m,n,o,grad,3,3);
	// cout<<"\nj11\n";
	// for (int i = 0; i < m*n*o; i++)
    // 	cout << J11[i] << " ";
	// cout<<"\nj12\n";
	// for (int i = 0; i < m*n*o; i++)
    // 	cout << J12[i] << " ";
	// cout<<"\nj13\n";
	// for (int i = 0; i < m*n*o; i++)
    // 	cout << J13[i] << " ";
	for(i=0;i<(m*n*o);i++){
		J11[i]*=factor1;
		J12[i]*=factor1;
		J13[i]*=factor1;
		J21[i]*=factor1;
		J22[i]*=factor1;
		J23[i]*=factor1;
		J31[i]*=factor1;
		J32[i]*=factor1;
		J33[i]*=factor1;
	}

	for(i=0;i<(m*n*o);i++){
		J11[i]+=1.0;
		J22[i]+=1.0;
		J33[i]+=1.0;
	}
	for(i=0;i<(m*n*o);i++){
		J=
		J11[i]*(J22[i]*J33[i]-J23[i]*J32[i])
		-J21[i]*(J12[i]*J33[i]-J13[i]*J32[i])
		+J31[i]*(J12[i]*J23[i]-J13[i]*J22[i]);

		jmean+=J;
		if(J>Jmax)
			Jmax=J;
		if(J<Jmin)
			Jmin=J;
		if(J<0)
			neg++;
		count++;
		Jac[i]=J;
	}
	jmean/=(m*n*o);
	for(int i=0;i<m*n*o;i++){
		jstd+=pow(Jac[i]-jmean,2.0);
	}
	jstd/=(m*n*o-1);
	jstd=sqrt(jstd);
	frac=neg/count;
	cout<<"mean(J)="<<jmean<<" ";
	cout<<"std(J)="<<jstd<<" ";
	//cout<<"Range: ["<<Jmin<<", "<<Jmax<<"]round(jmean*100)/100.0<<
    cout<<"(J<0)="<<round(frac*1e7)/100.0<<"e-7"<<" ";
	delete[] Jac;


	delete[] J11;
	delete[] J12;
	delete[] J13;
	delete[] J21;
	delete[] J22;
	delete[] J23;
	delete[] J31;
	delete[] J32;
	delete[] J33;

	return jstd;


}



void consistentMappingCL(float* u,float* v,float* w,float* u2,float* v2,float* w2,int m,int n,int o,int factor){
    float factor1=1.0/(float)factor;
    float* us=new float[m*n*o];
    float* vs=new float[m*n*o];
    float* ws=new float[m*n*o];
    float* us2=new float[m*n*o];
    float* vs2=new float[m*n*o];
    float* ws2=new float[m*n*o];

    for(int i=0;i<m*n*o;i++){
        us[i]=u[i]*factor1;
		vs[i]=v[i]*factor1;
		ws[i]=w[i]*factor1;

		us2[i]=u2[i]*factor1;
		vs2[i]=v2[i]*factor1;
		ws2[i]=w2[i]*factor1;
    }

    for(int it=0;it<10;it++){
        interp3(u,us2, us,vs,ws, m,n,o, m,n,o, true);
        interp3(v,vs2, us,vs,ws, m,n,o, m,n,o, true);
        interp3(w,ws2, us,vs,ws, m,n,o, m,n,o, true);

		for(int i=0;i<m*n*o;i++){
            u[i]=0.5*us[i]-0.5*u[i];
            v[i]=0.5*vs[i]-0.5*v[i];
            w[i]=0.5*ws[i]-0.5*w[i];

        }

		interp3(u2,us, us2,vs2,ws2, m,n,o, m,n,o, true);
        interp3(v2,vs, us2,vs2,ws2, m,n,o, m,n,o, true);
        interp3(w2,ws, us2,vs2,ws2, m,n,o, m,n,o, true);

		for(int i=0;i<m*n*o;i++){
            u2[i]=0.5*us2[i]-0.5*u2[i];
            v2[i]=0.5*vs2[i]-0.5*v2[i];
            w2[i]=0.5*ws2[i]-0.5*w2[i];
        }

        for(int i=0;i<m*n*o;i++){
            us[i]=u[i];
			vs[i]=v[i];
			ws[i]=w[i];

			us2[i]=u2[i];
			vs2[i]=v2[i];
			ws2[i]=w2[i];
        }
			// std::cout<<"Iter: "<<it<<"\n";
			// for(int pri=0;pri<m*n*o ;pri++){
			// 	std::cout<<u[pri]<<" ";
			// }
			// std::cout<<"\n";
    }


    for(int i=0;i<m*n*o;i++){
        u[i]*=(float)factor;
        v[i]*=(float)factor;
        w[i]*=(float)factor;
        u2[i]*=(float)factor;
        v2[i]*=(float)factor;
        w2[i]*=(float)factor;
    }


    delete[] us; delete[] vs; delete[] ws;
    delete[] us2; delete[] vs2; delete[] ws2;
}


void upsampleDeformationsCL(float* u_out,float* v_out,float* w_out, //full size flow field
							float* u_in,float* v_in,float* w_in, //gridded flow field: x-disps, y-disps, z-disps
							int m_out,int n_out,int o_out, //full size output
							int m_in,int n_in,int o_in){ //gridded size


    float scale_m=(float)m_out/(float)m_in; //full_size/gridded_size x (>1)
    float scale_n=(float)n_out/(float)n_in; //full_size/gridded_size y (>1)
    float scale_o=(float)o_out/(float)o_in; //full_size/gridded_size z (>1)

    float* x1=new float[m_out*n_out*o_out]; // full sized
    float* y1=new float[m_out*n_out*o_out];
    float* z1=new float[m_out*n_out*o_out];

    for(int k=0;k<o_out;k++){
        for(int j=0;j<n_out;j++){
            for(int i=0;i<m_out;i++){
                x1[i+j*m_out+k*m_out*n_out]=j/scale_n; //x helper var -> stretching factor in x-dir (gridded_size/full_size) at every discrete x (full size)
                y1[i+j*m_out+k*m_out*n_out]=i/scale_m; //y helper var
                z1[i+j*m_out+k*m_out*n_out]=k/scale_o; //z helper var
            }
        }
    }

    interp3(u_out, u_in, x1, y1, z1, m_out, n_out, o_out, m_in, n_in, o_in, false); //interpolate x dir, u1 is returned
    interp3(v_out, v_in, x1, y1, z1, m_out, n_out, o_out, m_in, n_in, o_in, false); //interpolate y dir, v1 is returned
    interp3(w_out, w_in, x1, y1, z1, m_out, n_out, o_out, m_in, n_in, o_in, false); //interpolate z dir, w1 is returned

    delete[] x1;
    delete[] y1;
    delete[] z1;
}


// libtorch unittest API
torch::Tensor transformations_jacobian(
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


torch::Tensor transformations_interp3(
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

torch::Tensor transformations_volfilter(
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
        torch::Tensor > transformations_consistentMappingCL(

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
        torch::Tensor> transformations_upsampleDeformationsCL(
            torch::Tensor pInput_u_in,
            torch::Tensor pInput_v_in,
            torch::Tensor pInput_w_in,
            torch::Tensor pOutput_size) {

    int m_in = pInput_u_in.size(0);
    int n_in = pInput_u_in.size(1);
    int o_in = pInput_u_in.size(2);

    int m_out = pOutput_size[0].item<int>();
    int n_out = pOutput_size[1].item<int>();
    int o_out = pOutput_size[2].item<int>();

    torch::Tensor input_u_in_copy = pInput_u_in.clone();
    torch::Tensor input_v_in_copy = pInput_v_in.clone();
    torch::Tensor input_w_in_copy = pInput_w_in.clone();

    float* u_in = input_u_in_copy.data_ptr<float>();
    float* v_in = input_v_in_copy.data_ptr<float>();
    float* w_in = input_w_in_copy.data_ptr<float>();

    float* u_out = new float[m_out*n_out*o_out];
    float* v_out = new float[m_out*n_out*o_out];
    float* w_out = new float[m_out*n_out*o_out];
    // cout<<"m"<<m;
    // cout<<"n"<<n;
    // cout<<"o"<<o;

    upsampleDeformationsCL(u_out, v_out, w_out, u_in, v_in, w_in,
        m_out, n_out, o_out, m_in, n_in, o_in);

    std::vector<float> new_u{u_out, u_out + m_out*n_out*o_out};
    std::vector<float> new_v{v_out, v_out + m_out*n_out*o_out};
    std::vector<float> new_w{w_out, w_out + m_out*n_out*o_out};

    auto options = torch::TensorOptions();

    return std::tuple<
            torch::Tensor,
            torch::Tensor,
            torch::Tensor>(
        torch::from_blob(new_u.data(), {m_out,n_out,o_out}, options).clone(),
        torch::from_blob(new_v.data(), {m_out,n_out,o_out}, options).clone(),
        torch::from_blob(new_w.data(), {m_out,n_out,o_out}, options).clone()
    );
}