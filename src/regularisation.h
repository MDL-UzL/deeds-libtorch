/* Incremental diffusion regularisation of parametrised transformation
 using (globally optimal) belief-propagation on minimum spanning tree.
 Fast distance transform uses squared differences.
 Similarity cost for each node and label has to be given as input.
*/
void messageDT(int ind,float* data,short* indout,int len1,float offsetx,float offsety,float offsetz){
    //for every possible parent displacement of a child, build up a lookup table which contains min cost and moving index of the child.
    //This will answer the question: When a parent moves e.g. one step to the right, what would be the optimal movement for the children be.
    //optimal movement of children depends on distance to parent (the higher the less optimal) and the own datacost (image intensity fitting)

    //input index of current child, patchcost data, search window width len1 and offset difference to parent node normalized by dilation value quant
    //int ind1=get_global_id(0)+start;
    //int ind=ordered[ind1];

    int len2=len1*len1;
    int len3=len1*len1*len1; //duplication
    float z[len1*2+1]; //len1 == hw*2+1



    float* val;
    float* valout;
    short* indo;

    float* valb;
    float* valb2;

    float buffer[len3]; //buffer of search window size (3D) = (2*hw+1)^3
    float buffer2[len3];
    int* indb;
    int* indb2;
    int bufferi[len3];
    int bufferi2[len3];


    //y-dir
    for(int i=0;i<len1*2+1;i++){
        // for double search window width + 1
        z[i]=(i - len1 + offsety) * (i - len1 + offsety);
        //parabola centered at +(len1-offsety) == +(2*hw+1-offsety)
        //build a quadratic offset model
        //the minimum value is 0 for z[len1-offsety]
        //offset is the value of the current active displacement of bigger patches from the last level (which include the smaller patches in the current iteration)
        //offset_y compensation makes sure, that the total displacement of patch acummulated over different levels is considered and not just the actual displacement
    }

    for(int k1=0;k1<len1;k1++){
        //for every step within search distance z-dir
        for(int j1=0;j1<len1;j1++){
            //for every step within search distance y-dir
            //valb=buffer2+(j1*len1+k1*len1*len1);//
            int num = (j1*len1 + k1*len1*len1); //storing index for linear-min-convolution value, index of y-datacost line
            val = data + ind*len3 + num; // we have (2*hw+1)^3 * patch_count size here, val is datacost val of child
            // in childrens space this is the datacost of the children D(f_p) messages from p->q
            // get datacost values of search position (around child index) with current z,y offset
            // (not y offsetted, it will be applied in the loop)
            // we hava a "line" of y-datacost values here

            valb2 = buffer + num;
             // we hava a "line" of buffered y-datacost values here
            indb = bufferi + num; //whats the minimum cost index when moving along that y-line?

            for(int i=0;i<len1;i++){
                // iterate over parents displacement (i.e. f_q label)
                float minval = val[0] + z[i+len1]; // D(f_p) + V(i -(j=0)) // this is just an initialization, could also be +inf?
                int minind=0;
                for(int j=0;j<len1;j++){
                    // iterate over child displacement (i.e. f_p label)
                    int p_c_displacement_diff = i-j;
                    // parent may move +hw, children may move -hw which results in diff 2*hw in either direction
                    // if displacement direction between parent and children is the same do not add V(i-j) = z[i-j]
                    bool b = (val[j] + z[p_c_displacement_diff + len1] < minval);

                    //find minimum value and corresponding index
                    minval = b ? val[j]+z[p_c_displacement_diff+len1] : minval; // min-convolution (datacost value at search position y + some parabola value at fighting positions i,j? offseted by 2*hw+1)
                    minind = b ? j : minind; //minimum index is going to be the search position / offset j i.e. y
                }
                //update this value for every parent displacement f_q -> for a parents displacement, whats the minimum moving y direction for the child?
                valb2[i] = minval; //update buffer
                indb[i] = num + minind; //upate bufferi, parents
            }

        }
    }
    //x-dir
    for(int i=0;i<len1*2;i++){
        z[i]=(i-len1+offsetx)*(i-len1+offsetx);
    }
    for(int k1=0;k1<len1;k1++){
        for(int i1=0;i1<len1;i1++){
            valb=buffer+(i1+k1*len1*len1);
            valb2=buffer2+(i1+k1*len1*len1);
            indb=bufferi+(i1+k1*len1*len1);
            indb2=bufferi2+(i1+k1*len1*len1);

            for(int i=0;i<len1;i++){
                float minval=valb[0]+z[i+len1];
                int minind=0;
                for(int j=0;j<len1;j++){
                    bool b=(valb[j*len1]+z[i-j+len1]<minval);
                    minval=b?valb[j*len1]+z[i-j+len1]:minval;
                    minind=b?j:minind; //update x index value

                }
                //update along x
                valb2[i*len1]=minval; //update buffer2
                indb2[i*len1]=indb[minind*len1]; //read bufferi and update bufferi2, steps in x-dir
            }

        }
    }
    //z-dir
    for(int i=0;i<len1*2;i++){
        z[i]=(i-len1+offsetz)*(i-len1+offsetz);

    }
    for(int j1=0;j1<len1;j1++){
        for(int i1=0;i1<len1;i1++){
            valb=buffer2+(i1+j1*len1);
            //valb2=buffer+(i1+j1*len1);
            valout=data+ind*len3+(i1+j1*len1);
            indb=bufferi2+(i1+j1*len1);
            //indb2=bufferi+(i1+j1*len1);
            indo=indout+ind*len3+(i1+j1*len1);
            for(int i=0;i<len1;i++){
                float minval=valb[0]+z[i+len1];
                int minind=0;
                for(int j=0;j<len1;j++){
                    bool b=(valb[j*len2]+z[i-j+len1]<minval);
                    minval=b?valb[j*len2]+z[i-j+len1]:minval;
                    minind=b?j:minind;
                }
                valout[i*len2]=minval;
                indo[i*len2]=indb[minind*len2];
                //update z index value, update indout with bufferi2 data
            }
        }
    }





}

void regularisationCL(float* costall,float* u0,float* v0,float* w0,float* u1,float* v1,float* w1,int hw,int step1,float quant,int* ordered,int* parents,float* edgemst)
{
    //Input: patch cost costall, initial displacements u0,v0,w0, search window hw, grid step step1, dilation factor quant, ordered vertex indices of graph (the higher the indices the higher the level)
    //input: edgemst edgemessages
    //Output: displacements u1,v1,w1

	int m2=image_m;
	int n2=image_n;
	int o2=image_o;

	int m=m2/step1;
	int n=n2/step1;
	int o=o2/step1;

	timeval time1,time2;

	int sz=m*n*o; //patch count
    int len=hw*2+1;
    int len1=len;
	int len2=len*len*len;
    int len3=len*len*len;

    gettimeofday(&time1, NULL);


	short *allinds=new short[sz*len2]; // here all relative optimal displacement positions for a given parent displacement will be stored.
	float *cost1=new float[len2];
	float *vals=new float[len2];
	int *inds=new int[len2];


    //calculate level boundaries for parallel implementation
    int* levels=new int[sz];
    for(int i=0;i<sz;i++){
        levels[i]=0;
    }
    for(int i=1;i<sz;i++){
        int ochild=ordered[i];
		int oparent=parents[ordered[i]];
        levels[ochild]=levels[oparent]+1; //set level of vertices (couldnt that be retrieved by levels[ochild]?)
    }
    int maxlev=1+*max_element(levels,levels+sz);

    int* numlev=new int[maxlev];

    int* startlev=new int[maxlev];
    for(int i=0;i<maxlev;i++){
        numlev[i]=0;
    }
    for(int i=0;i<sz;i++){
        numlev[levels[i]]++;
    }
    startlev[0]=numlev[0];
    for(int i=1;i<maxlev;i++){ //cumulative sum -> same as in primsMST (vertex index at which a level "starts" e.g. if 40 vertices are in level 1,2,3 offset is 40 for level 4)
        startlev[i]=startlev[i-1]+numlev[i];
    }
    delete[] levels;

	int xs1,ys1,zs1,xx,yy,zz,xx2,yy2,zz2;

	for(int i=0;i<len2;i++){
		cost1[i]=0;
	}

    //MAIN LOOP - TO BE PARALLELISED
	int frac=(int)(sz/25);
    int counti=0;
    int counti2=0;

    bool* processed=new bool[sz];
    for(int i=0;i<sz;i++){
        processed[i]=false; //unused
    }
    int dblcount=0;
    float timeCopy=0;
    float timeMessage=0;
	//calculate mst-cost
    for(int lev=maxlev-1;lev>0;lev--){
        // iterate over level vertices starting at high levels (deep in the tree)
        int start=startlev[lev-1];
        int length=numlev[lev];


        gettimeofday(&time1, NULL);

        for(int i=start;i<start+length;i++){
            int ochild=ordered[i];
            for(int l=0;l<len2;l++){
                costall[ochild*len2+l]*=edgemst[ochild]; // multiply with passed message for every child in level?
            }
        }
#pragma omp parallel for
        for(int i=start;i<start+length;i++){
            //for every node in level
            int ochild=ordered[i];
            int oparent=parents[ordered[i]];

            float offsetx=(u0[oparent]-u0[ochild])/(float)quant; // this is the displacement difference of parent and chilren normalized by dilation factor
            float offsety=(v0[oparent]-v0[ochild])/(float)quant;
            float offsetz=(w0[oparent]-w0[ochild])/(float)quant;
            messageDT(ochild,costall,allinds,len1,offsetx,offsety,offsetz); // get allinds (inds for minimum value reached?)
        }


        gettimeofday(&time2, NULL);
        timeMessage+=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);

        gettimeofday(&time1, NULL);
        // above and beyond loops are only split due to time measurements.

        //copy necessary if vectorisation is used (otherwise multiple simultaneous +='s)
        int start0=startlev[lev-1];
        int length0=numlev[lev];
        for(int i=start0;i<start0+length0;i++){
            //for every node in level
            int ochild=ordered[i];
            int oparent=parents[ordered[i]];
            float minval = *min_element(costall + (ochild*len2), costall + (ochild*len2) + len3); //len2 == len3 == (2*hw+1)^3 == search cube size == number of possible displacements of patch! -> mincost for displacement within the number of possible displacements

            for(int l=0;l<len2;l++){
                //iterate over every position in search window
                costall[oparent*len2+l] += (costall[ochild*len2+l]-minval);///edgemst[ochild];//transp //reduce all patch costs around search position of children by minimum value of patch cost within that search cube (for one position 0 will be added to costall of parent)
                //if a parent moves in a certain direction, what would be the cost for the children moving in the same direction? -> store the sum over all chilren of a parent inside the parent.
                //pass cost of childrens to cost of parent
                //edgemst[ochild]*

            }
        }

        gettimeofday(&time2, NULL);
        timeCopy+=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);


    }


    //dense displacement space
	float* xs=new float[len*len*len];
	float* ys=new float[len*len*len];
	float* zs=new float[len*len*len];

	for(int i=0;i<len;i++){
		for(int j=0;j<len;j++){
			for(int k=0;k<len;k++){
				xs[i+j*len+k*len*len]=(j-hw)*quant; // this is a fixed initialization (direction and distance of displacements in voxel space (may have been converte to search window space before by quant/dilation))
				ys[i+j*len+k*len*len]=(i-hw)*quant; // this is a fixed initialization
				zs[i+j*len+k*len*len]=(k-hw)*quant; // this is a fixed initialization
			}
		}
	}
    //xs, ys, zs servere as a lookup map. a passed index of the search cube is translated to voxel offsets here.

    int *selected=new int[sz];

	//mst-cost & select displacement for root note
	int i=0;
	int oroot=ordered[i];
	for(int l=0;l<len2;l++){
        cost1[l]=costall[oroot*len2+l];//transp

	}
	float value=cost1[0];
    int index=0;

	for(int l=0;l<len2;l++){
		if(cost1[l]<value){
			value=cost1[l]; //store minimum cost for root node
			index=l;
		}
        allinds[oroot*len2+l]=l; //transp

	}
	selected[oroot]=index; // set search index l (i.e. displacement index) for patch index of root
	u1[oroot]=xs[index]+u0[oroot]; // add initial displacement of root to displacement position with minimized cost value.
	v1[oroot]=ys[index]+v0[oroot];
	w1[oroot]=zs[index]+w0[oroot];


	//select displacements and add to previous deformation field
	for(int i=1;i<sz;i++){
        //for all nodes (except root)
		int ochild=ordered[i];
		int oparent=parents[ordered[i]];
		//select from argmin of based on parent selection
		//index=allinds[ochild+selected[oparent]*sz];
        index=allinds[ochild*len2+selected[oparent]]; //transp // select the optimal displacement of a child when the parent is displaced by selected[parent]
		selected[ochild]=index; // calculate here how the children should move. This info is provided by the messageDT function
		u1[ochild]=xs[index]+u0[ochild];
		v1[ochild]=ys[index]+v0[ochild];
		w1[ochild]=zs[index]+w0[ochild];

	}

	//cout<<"Deformation field calculated!\n";

	delete[] cost1;
	delete[] vals;
	delete[] inds;
	delete[] allinds;
	delete[] selected;

}
