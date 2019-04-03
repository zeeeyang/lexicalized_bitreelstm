#include "cnn/bitreelstm.h"
#include "cnn/dict.h"
#include "cnn/training.h"
#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/timing.h"
#include "cnn/expr.h"
#include <iostream>
#include <fstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <unordered_map>
#include <unordered_set>
using namespace cnn;
using namespace cnn::expr;
using namespace std;

unsigned LAYERS = 1;
unsigned INPUT_DIM = 64;
unsigned HIDDEN_DIM = 150;//75;//150;
unsigned TAG_HIDDEN_DIM = 128;
unsigned LABEL_SIZE = 5;
int kUNK; //tzy
float pdrop = 0.5;
bool eval = false;
bool verbose = false;


unordered_map<string, vector<float> > pretrained_embeddings;
unordered_set<string> training_set;

vector<float> averaged_vec;
cnn::Dict word_dict;

void ReadEmbeddings(const string& input_file)
{
    ifstream fin(input_file);
    assert(fin);
    string line;
    bool first = true;
    while(getline(fin, line))
    {
        istringstream sin(line);
        string word;
        sin>>word;
        float value;
        vector<float> vecs;
        while(sin>>value)
            vecs.push_back(value);
        if(first)
        {
            first = false;
            INPUT_DIM = vecs.size();
            averaged_vec.resize(INPUT_DIM, 0.0);
        }
        for(int i = 0; i< INPUT_DIM; i++)
        {
            averaged_vec[i] += vecs[i];
        }
        pretrained_embeddings[word] = vecs;
        word_dict.Convert(word); //new
        //training_set.insert(word);
    }
    fin.close();
    for(int i = 0; i< INPUT_DIM; i++)
    {
        averaged_vec[i] /= pretrained_embeddings.size();
    }
}

void ReadCorpus(const string& inputPath, vector<BiTree*>& trees)
{
    ifstream in(inputPath);
    assert(in);
    string line;
    while(getline(in, line)) {
        //cerr<< line << endl;
        BiTree* biTree = new BiTree;
        biTree->accept(line);
        //biTree->printWords();
        trees.push_back(biTree);
        //trees.back()->printWords();
    }
    in.close();
    //cerr << "lines: " << trees.size() << endl;
}

void Clear(vector<BiTree*>& trees)
{
    for(auto& tree: trees)
        delete tree;
    trees.clear();
}

int main(int argc, char** argv)
{
    //cnn::Initialize(argc, argv, 2185744987);
    cnn::Initialize(argc, argv);

    if (argc != 5 && argc != 6) {
        cerr << "Usage: " << argv[0] << " vec train.tree dev.tree test.tree [model.params]\n";
        return 1;
    }
    cerr << "Reading pretrained vector from " << argv[1] << "...\n";
    ReadEmbeddings(argv[1]);

    vector<BiTree*> training_trees, dev_trees, test_trees;
    cerr << "Reading training data from " << argv[2] << "...\n";
    ReadCorpus(argv[2], training_trees);
    unordered_map<string, int> word_counts;
    for(auto& tree: training_trees)
        tree->countWords(word_counts);


    cerr << "Reading dev data from " << argv[3] << "...\n";
    ReadCorpus(argv[3], dev_trees);

    cerr << "Reading test data from " << argv[4] << "...\n";
    ReadCorpus(argv[4], test_trees);
    kUNK = word_dict.Convert("<|unk|>");
    //since vocabulary cannot be enlarged automatically, so this will be have no affect to training
    //set unk
    for(auto& tree: training_trees)
        tree->setUnk(word_dict, kUNK, pretrained_embeddings, word_counts, training_set, true);
    word_dict.Freeze();
    word_dict.SetUnk("<|unk|>");
    cerr<<"DEV UNK"<<endl;
    for(auto& tree: dev_trees)
        tree->setUnk(word_dict, kUNK, pretrained_embeddings, word_counts, training_set, false);
    cerr<<"TEST UNK"<<endl;
    for(auto& tree: test_trees)
        tree->setUnk(word_dict, kUNK, pretrained_embeddings, word_counts, training_set, false);
    //for(auto& tree: dev_trees)
    //    tree->printWords();
    unsigned VOCAB_SIZE = word_dict.size();
    cerr<< "Vocab Size: " << VOCAB_SIZE << endl;
    //exit(0);

    ostringstream os;
    os << "bitree.zhu.fix2."
       << '_' << INPUT_DIM
       << '_' << HIDDEN_DIM
       << '_' << LAYERS
       << "-pid" << getpid()
       << ".params";
    const string fname = os.str();
    cerr << "Parameters will be written to: " << fname << endl;
    double best = 0;

    Model model;
    bool use_momentum = true;
    Trainer* sgd = nullptr;
    if (use_momentum)
        //sgd = new MomentumSGDTrainer(&model, 1e-4, 0.05);
        //sgd = new MomentumSGDTrainer(&model,1e-6, 0.1);
        //sgd = new MomentumSGDTrainer(&model);
        //sgd = new AdagradTrainer(&model);
        sgd = new AdamTrainer(&model);
    else
        sgd = new SimpleSGDTrainer(&model);
    //sgd->eta_decay = 0.08; //2016-02-21
    //sgd->eta_decay = 0.05; //2016-02-21
    //sgd->eta_decay = 0.1; //2016-02-21
    sgd->clipping_enabled = false;

    //biTree.printWords();
    BiTreeBuilder treeBuilder(LAYERS,
                              INPUT_DIM,
                              HIDDEN_DIM,
                              VOCAB_SIZE,
                              TAG_HIDDEN_DIM,
                              LABEL_SIZE,
                              kUNK,
                              &model,
                              pdrop);

    treeBuilder.initEmbeddings(VOCAB_SIZE, word_dict, pretrained_embeddings, averaged_vec);

    if(argc == 6)
    {
        string fname = argv[5];
        ifstream in(fname);
        boost::archive::text_iarchive ia(in);
        ia >> model;
        int drcorr = 0;
        int dpcorr = 0;
        float dloss = 0;
        int dtotal  = 0;
        eval = true;
        //verbose = true;
        for (auto& biTree : dev_trees) {
            ComputationGraph cg;
            treeBuilder.buildGraph(cg, *biTree, eval, verbose);
            //biTree->printWords();
            Expression yloss = treeBuilder.softmax(cg, *biTree, dtotal, drcorr, dpcorr, eval, verbose);
            dloss += as_scalar(cg.incremental_forward());
        }
        cerr<<"DEV Loss:"<< dloss << endl;
        cerr<<"DEV Root Accuracy:"<< drcorr <<"/" << dev_trees.size()<<" "<< ((float)drcorr)/dev_trees.size() << endl;
        cerr<<"DEV Phrase Accuracy:"<< dpcorr <<"/" << dtotal <<" "<< ((float)dpcorr)/dtotal << endl;

        int trecorr = 0;
        int tpecorr = 0;
        float teloss = 0;
        int tetotal  = 0;
        //verbose = true;
        for (auto& biTree : test_trees) {
            ComputationGraph cg;
            treeBuilder.buildGraph(cg, *biTree, eval, verbose);
            //biTree->printWords();
            Expression yloss = treeBuilder.softmax(cg, *biTree, tetotal, trecorr, tpecorr, eval, verbose);
            teloss += as_scalar(cg.incremental_forward());
        }
        cerr<<"Test Loss:"<< teloss << endl;
        cerr<<"Test Root Accuracy:"<< trecorr <<"/" << test_trees.size()<<" "<< ((float)trecorr)/test_trees.size() << endl;
        cerr<<"Test Phrase Accuracy:"<< tpecorr <<"/" << tetotal <<" "<< ((float)tpecorr)/tetotal << endl;
        return 0;
    }

    unsigned report_every_i = min(100, int(training_trees.size()));
    unsigned si = training_trees.size();
    vector<unsigned> order(training_trees.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
    bool first = true;
    int report = 0;
    unsigned lines = 0;
    int trcorr = 0;
    int tpcorr = 0;
    int ttotal = 0;
    int count = 0;
    while(count <= 30) {
        Timer iteration("completed in");
        double loss = 0;
        for (unsigned i = 0; i < report_every_i; ++i) {
            if (si == training_trees.size()) {
                count++;
                si = 0;
                trcorr = 0;
                tpcorr = 0;
                ttotal = 0;
                if (first) {
                    first = false;
                }
                else {
                    //report++;
                    if (1) {
                        double dloss = 0;
                        int drcorr = 0;
                        int dpcorr = 0;
                        int dtotal = 0;
                        eval = true;
                        //verbose = true;
                        cerr<<endl;

                        //cerr<< "DEV LOG" << endl;
                        for (auto& biTree : dev_trees) {
                            //cerr<< "[begin]: "<< eval << "\t"<< verbose << endl;
                            ComputationGraph cg;
                            treeBuilder.buildGraph(cg, *biTree, eval, verbose);
                            //biTree->printWords();
                            //cerr<< "[end]: "<< eval << "\t"<< verbose << endl;
                            Expression yloss = treeBuilder.softmax(cg, *biTree, dtotal, drcorr, dpcorr, eval, verbose);
                            dloss += as_scalar(cg.incremental_forward());
                            //break;
                        }
                        double acc = drcorr/(float)dev_trees.size();
                        //double acc = dpcorr/(float)dtotal + drcorr/(float)dev_trees.size();
                        //double acc =  drcorr/(float)dev_trees.size();
                        cerr << "\n***DEV [epoch=" << (lines / (double)training_trees.size()) << "] E = " << dloss << " acc=" << acc;
                        cerr << " \troot acc=" << drcorr/(float)dev_trees.size() << ' '<< drcorr << "/"<< dev_trees.size()<<endl;
                        //cerr<< " \tphrase acc="<< dpcorr/(float)dtotal << " "<< dpcorr << "/"<< dtotal<<endl;
                        if (acc >= best) {
                            cerr<< endl<<"Exceed" << endl;
                            best = acc;
                            ofstream out(fname);
                            boost::archive::text_oarchive oa(out);
                            oa << model;
                            int trecorr = 0;
                            int tpecorr = 0;
                            float teloss = 0;
                            int tetotal  = 0;
                            //verbose = true;
                            for (auto& biTree : test_trees) {
                                ComputationGraph cg;
                                treeBuilder.buildGraph(cg, *biTree, eval, verbose);
                                //biTree->printWords();
                                Expression yloss = treeBuilder.softmax(cg, *biTree, tetotal, trecorr, tpecorr, eval, verbose);
                                teloss += as_scalar(cg.incremental_forward());
                            }
                            cerr<<"Test Loss:"<< teloss << endl;
                            cerr<<"Test Root Accuracy:"<< trecorr <<"/" << test_trees.size()<<" "<< ((float)trecorr)/test_trees.size() << endl;
                            //cerr<<"Test Phrase Accuracy:"<< tpecorr <<"/" << tetotal <<" "<< ((float)tpecorr)/tetotal << endl;
                        }
                    }
                    eval = false;
                    verbose = false;

                    sgd->update_epoch();
                }
                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);
            }
            // build graph for this instance
            ComputationGraph cg;
            auto& biTree = training_trees[order[si]];
            treeBuilder.buildGraph(cg, *biTree, eval, verbose);
            //cerr << "LINE: " << order[si] << endl;
            Expression yloss = treeBuilder.softmax(cg, *biTree, ttotal, trcorr, tpcorr, eval, verbose);
            loss += as_scalar(cg.incremental_forward());
            cg.backward();
            sgd->update(1.0);
            ++si;
            ++lines;
        }
        sgd->status();
        cerr << " E = " << loss  <<" ";
        // show score on dev data?
    }
    delete sgd;
    Clear(training_trees);
    Clear(dev_trees);
    Clear(test_trees);
    return 0;
}
