### Root Model 
Model for Fine-grained Sentiment Classification. 
### Phrase Model
Model for Binary Sentiment Classification. 
### Compile
First, enter the corresponding model folder, for example, 
```
cd root_model
```
Second, change the eigen path in the `cmake.sh`, 
```
cmake . -DEIGEN3_INCLUDE_DIR=<your-eigen-path-here>
```
Then, just run `./cmake.sh` to compile the corresponding models. 

Once `BiTreeSentimentZhu` is generated, the project is successfully compiled. 
