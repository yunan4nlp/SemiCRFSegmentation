#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	Alphabet sparseAlpha;// should be initialized outside

	SparseParams sparse_project; //sparse


public:
	Alphabet labelAlpha; // should be initialized outside
	CRFMLLoss loss;


public:
	bool initial(HyperParams& opts, AlignedMemoryPool* mem){

		// some model parameters should be initialized outside
		if (labelAlpha.size() <= 0){
			return false;
		}
		opts.labelSize = labelAlpha.size();
		sparse_project.initial(&sparseAlpha, opts.labelSize);

		loss.initial(opts.labelSize);

		return true;
	}


	bool directInitial(HyperParams& opts, AlignedMemoryPool* mem){

		// some model parameters should be initialized outside
		if (labelAlpha.size() <= 0){
			return false;
		}
		opts.labelSize = labelAlpha.size();
		sparse_project.initial(&sparseAlpha, opts.labelSize);
		return true;
	}



	void exportModelParams(ModelUpdate& ada){
		sparse_project.exportAdaParams(ada);
		loss.exportAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
	}


	void saveModel(ofstream& os) const{
		sparseAlpha.write(os);

		labelAlpha.write(os); 
		loss.T.save(os);
		os << loss.labelSize << std::endl;
		int buffer_size = loss.buffer.size();
		os << buffer_size << std::endl;
		for (int idx = 0; idx < buffer_size; idx++){
			os << loss.buffer[idx] << std::endl;
		}
		os << loss.eps << std::endl;

	}

	void loadModel(ifstream& is, AlignedMemoryPool* mem = NULL){
		sparseAlpha.read(is);

		labelAlpha.read(is); 

		loss.T.load(is);
		is >> loss.labelSize;
		int buffer_size;
		is >> buffer_size;
		loss.buffer.resize(buffer_size);
		for (int idx = 0; idx < buffer_size; idx++){
			is >> loss.buffer[idx];
		}
		is >> loss.eps;
	}

};

#endif /* SRC_ModelParams_H_ */