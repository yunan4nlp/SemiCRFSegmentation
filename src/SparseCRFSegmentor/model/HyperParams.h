#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3L.h"
#include "Example.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams{

	// must assign
	int wordcontext;
	int hiddensize;
	int rnnhiddensize;
	dtype dropOut;


	dtype nnRegular; // for optimization
	dtype adaAlpha;  // for optimization
	dtype adaEps; // for optimization



	//auto generated
	int wordwindow;
	int wordDim;
	vector<int> typeDims;
	int unitsize;
	int inputsize;
	int labelSize;

public:
	HyperParams(){
		bAssigned = false;
	}

public:
	void setRequared(Options& opt){
		wordcontext = opt.wordcontext;
		hiddensize = opt.hiddenSize;
		rnnhiddensize = opt.rnnHiddenSize;
		dropOut = opt.dropProb;
		nnRegular = opt.regParameter;
		adaAlpha = opt.adaAlpha;
		adaEps = opt.adaEps;

		bAssigned = true;
	}

	void clear(){
		bAssigned = false;
	}

	bool bValid(){
		return bAssigned;
	}

	void saveModel(ofstream& os) const{
		os <<  wordcontext << std::endl;
		os <<  hiddensize << std::endl;
		os <<  rnnhiddensize << std::endl;
		os <<  dropOut << std::endl;


		os <<  nnRegular << std::endl; 
		os <<  adaAlpha << std::endl; 
		os <<  adaEps << std::endl; 

		os <<  wordwindow << std::endl;
		os <<  wordDim << std::endl;

		int typeDimsSize = typeDims.size();
		os << typeDimsSize << std::endl;
		for (int idx = 0; idx < typeDimsSize; idx++)
			os << typeDims[idx] << std::endl;

		os <<  unitsize << std::endl;
		os <<  inputsize << std::endl;
		os <<  labelSize << std::endl;
	}

	void loadModel(ifstream& is){
		is >>  wordcontext;
		is >>  hiddensize;
		is >>  rnnhiddensize;
		is >>  dropOut;

		is >>  nnRegular; 
		is >>  adaAlpha; 
		is >>  adaEps;

		is >>  wordwindow;
		is >>  wordDim;

		int typeDimsSize = typeDims.size();
		is >> typeDimsSize;
		typeDims.resize(typeDimsSize);
		for (int idx = 0; idx < typeDimsSize; idx++)
			is >> typeDims[idx];

		is >>  unitsize;
		is >>  inputsize;
		is >>  labelSize;
		bAssigned = true;
	}

public:

	void print(){

	}

private:
	bool bAssigned;
};


#endif /* SRC_HyperParams_H_ */