#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	Alphabet wordAlpha; // should be initialized outside
	LookupTable words; // should be initialized outside
	Alphabet charAlpha; // should be initialized outside
	LookupTable chars; // should be initialized outside
	Alphabet sparseAlpha;// should be initialized outside
	SparseParams sparse_project;

	vector<Alphabet> typeAlphas; // should be initialized outside
	vector<LookupTable> types;  // should be initialized outside


	LSTM1Params left_lstm_project; //left lstm
	LSTM1Params right_lstm_project; //right lstm
	UniParams tanh1_project; // hidden
	BiParams tanh2_project; // hidden
	UniParams tanh3_project; // output
	UniParams olayer_linear; // output


public:
	Alphabet labelAlpha; // should be initialized outside
	CRFMLLoss loss;


public:
	bool initial(HyperParams& opts, AlignedMemoryPool* mem){

		// some model parameters should be initialized outside
		if (words.nVSize <= 0 || labelAlpha.size() <= 0){
			return false;
		}
		opts.wordwindow = 2 * opts.wordcontext + 1;
		opts.wordDim = words.nDim;
		opts.charDim = chars.nDim;
		opts.unitsize = opts.wordDim;
		opts.typeDims.clear();
		for (int idx = 0; idx <types.size(); idx++){
			if (types[idx].nVSize <= 0 || typeAlphas[idx].size() <= 0){
				return false;
			}
			opts.typeDims.push_back(types[idx].nDim);
			opts.unitsize += opts.typeDims[idx];
		}
		opts.unitsize += opts.charDim * 3;
		opts.labelSize = labelAlpha.size();
		opts.inputsize = opts.wordwindow * opts.unitsize;

		tanh1_project.initial(opts.hiddensize, opts.inputsize, true, mem);
		left_lstm_project.initial(opts.rnnhiddensize, opts.hiddensize, mem);
		right_lstm_project.initial(opts.rnnhiddensize, opts.hiddensize, mem);
		tanh2_project.initial(opts.hiddensize, opts.rnnhiddensize, opts.rnnhiddensize, true, mem);
		tanh3_project.initial(opts.hiddensize, opts.hiddensize, true, mem);
		olayer_linear.initial(opts.labelSize, opts.hiddensize, false, mem);
		sparse_project.initial(&sparseAlpha, opts.labelSize);

		loss.initial(opts.labelSize);

		return true;
	}


	bool directInitial(HyperParams& opts, AlignedMemoryPool* mem){

		// some model parameters should be initialized outside
		if (words.nVSize <= 0 || labelAlpha.size() <= 0){
			return false;
		}
		opts.wordwindow = 2 * opts.wordcontext + 1;
		opts.wordDim = words.nDim;
		opts.charDim = chars.nDim;
		opts.unitsize = opts.wordDim;
		opts.typeDims.clear();
		for (int idx = 0; idx <types.size(); idx++){
			if (types[idx].nVSize <= 0 || typeAlphas[idx].size() <= 0){
				return false;
			}
			opts.typeDims.push_back(types[idx].nDim);
			opts.unitsize += opts.typeDims[idx];
		}
		opts.labelSize = labelAlpha.size();
		opts.inputsize = opts.wordwindow * opts.unitsize;
		sparse_project.initial(&sparseAlpha, opts.labelSize);
		return true;
	}



	void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
		chars.exportAdaParams(ada);
		for (int idx = 0; idx < types.size(); idx++){
			types[idx].exportAdaParams(ada);
		}
		tanh1_project.exportAdaParams(ada);
		left_lstm_project.exportAdaParams(ada);
		right_lstm_project.exportAdaParams(ada);
		tanh2_project.exportAdaParams(ada);
		tanh3_project.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
		sparse_project.exportAdaParams(ada);
		loss.exportAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&(words.E), "_words.E");
		for (int idx = 0; idx < types.size(); idx++){
			stringstream ss;
			ss << "types[" << idx << "].E";
			checkgrad.add(&(types[idx].E), ss.str());
		}
		checkgrad.add(&(tanh1_project.W), "tanh1_project.W");
		checkgrad.add(&(tanh1_project.b), "tanh1_project.b");

		checkgrad.add(&(tanh2_project.W1), "tanh2_project.W1");
		checkgrad.add(&(tanh2_project.W2), "tanh2_project.W2");
		checkgrad.add(&(tanh2_project.b), "tanh2_project.b");

		checkgrad.add(&(tanh3_project.W), "tanh3_project.W");
		checkgrad.add(&(tanh3_project.b), "tanh3_project.b");

		checkgrad.add(&(olayer_linear.W), "olayer_linear.W");
		checkgrad.add(&(loss.T), "loss.T");
	}


	void saveModel(ofstream& os) const{
		wordAlpha.write(os);
		words.save(os);
		charAlpha.write(os);
		chars.save(os);

		int type_alpha_size = typeAlphas.size();
		os << type_alpha_size << endl;
		for (int idx = 0; idx < type_alpha_size; idx++)
			typeAlphas[idx].write(os);

		int type_size = types.size();
		os << type_size << endl;
		for (int idx = 0; idx < type_size; idx++)
			types[idx].save(os);  


		left_lstm_project.save(os); 
		right_lstm_project.save(os); 
		tanh1_project.save(os); 
		tanh2_project.save(os); 
		tanh3_project.save(os); 
		olayer_linear.save(os); 
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
		wordAlpha.read(is);
		words.load(is, &wordAlpha, mem);
		charAlpha.read(is);
		chars.load(is, &charAlpha, mem);

		int type_alpha_size;
		is >> type_alpha_size;
		typeAlphas.resize(type_alpha_size);
		for (int idx = 0; idx < type_alpha_size; idx++)
			typeAlphas[idx].read(is);

		int type_size;
		is >> type_size;
		types.resize(type_size);
		for (int idx = 0; idx < type_size; idx++)
			types[idx].load(is, &typeAlphas[idx], mem);  


		left_lstm_project.load(is); 
		right_lstm_project.load(is); 
		tanh1_project.load(is); 
		tanh2_project.load(is); 
		tanh3_project.load(is); 
		olayer_linear.load(is); 
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