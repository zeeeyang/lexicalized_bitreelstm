/*************************************************************************
	> File Name: bitreelstm.h
	> Author:
	> Mail:
	> Created Time: Wed 20 Jan 2016 10:15:40 PM SGT
 ************************************************************************/

#ifndef _BITREELSTM_H
#define _BITREELSTM_H
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <unordered_set>

#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/timing.h"
#include "cnn/dict.h"

using namespace std;
using namespace cnn::expr;
using namespace cnn;

struct BiTreeNode {
    vector<struct BiTreeNode*> children;
    int label;
    int wordId;
    string word;
    Expression xi;
    bool isLeaf;
    bool isSingle;

    Expression td_hi;
    Expression td_ci;

    Expression bu_hi;
    Expression bu_ci;

    Expression output_td_hi;

    BiTreeNode(): isLeaf(false),isSingle(false)
    {
        children.clear();
    }
    ~BiTreeNode() {
        if(children.size() == 0) return;
        for(auto& child: children)
        {
            if(child !=NULL)
                delete child;
            child = NULL;
        }
    }
};

class BiTree {
public:
    struct BiTreeNode* root;
    BiTree():root(NULL) {}
    bool accept(const string& treerep);
    ~BiTree();
    void printWords();
    void printWords(vector<string>& words, struct BiTreeNode& root);
    void countWords(unordered_map<string, int>& word_counts);
    void setUnk(cnn::Dict& dict, int unkId, unordered_map<string, vector<float> >& pretrained_embeddings,unordered_map<string, int>& word_counts, unordered_set<string>& training_set, bool isTrain);
private:
    bool accept(const string& treerep, struct BiTreeNode& root);
    void countWords(unordered_map<string, int>& word_counts, struct BiTreeNode& root);
    void setUnk(struct BiTreeNode& root, cnn::Dict& dict, int unkId, unordered_map<string, vector<float> >& pretrained_embeddings, unordered_map<string, int>& word_counts, unordered_set<string>& training_set, bool isTrain);
};
/*
 * parameters of lstm
 */
class BiTreeBuilder
{
public:
    LookupParameters* p_word;
    Parameters* p_left2tag;
    Parameters* p_right2tag;
    Parameters* p_tagbias;

    Parameters* p_tag2label;
    Parameters* p_labelbias;

    vector<vector<Parameters*>> params;
    vector<vector<Expression>>  param_exprs;

    unsigned layers;
    float pdrop;
    int kUNK;

    BiTreeBuilder() = default;
    explicit BiTreeBuilder(unsigned layers,unsigned input_dim, unsigned hidden_dim,
                           unsigned vocab_size,
                           unsigned tag_hidden_dim,
                           unsigned label_size,
                           int kUNK,
                           Model* model, float pdrop=0.5);

    void initEmbeddings(unsigned vocab_size, cnn::Dict& dict, unordered_map<string, vector<float> > & pretrained_embeddings, const vector<float>& averaged_vec);

    void buildGraph(ComputationGraph& cg, BiTree& tree, bool eval=false, bool verbose=false);

    Expression softmax(ComputationGraph& cg, BiTree& tree, int& total, int& rcor, int& pcor, bool eval=false, bool verbose=false);

private:
    void buildExpression(ComputationGraph& cg);
    void buildBottomUpGraph(ComputationGraph& cg, struct BiTreeNode& tree, bool eval=false, bool verbose=false);
    void buildTopDownGraph(ComputationGraph& cg, struct BiTreeNode& tree, bool eval = false, bool verbose=false);

    vector<Expression> buildNodeRepresentation(ComputationGraph& cg, struct BiTreeNode& root, bool eval=false, bool verbose=false);

    vector<Expression> buildTopDownHiddenRepresentation(ComputationGraph& cg, struct BiTreeNode& root, bool eval=false, bool verbose=false);

    void buildTopDownGraph(ComputationGraph& cg, struct BiTreeNode& root, struct BiTreeNode& rootParent, bool isLeft, bool eval, bool verbose);
    void softmax(ComputationGraph& cg, BiTree& bi_tree, struct BiTreeNode& treeNode, vector<Expression>& loss, int& total, int& rcor, int& pcor, bool eval=false, bool verbose=false);

};

#endif
