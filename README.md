# The Avg-Act Swap
The Avg-Act Swap ('the Swap') is a technique to place the average pooling layer ('AvgPool') before an activation function ('Activation') in machine learning models over encrypted data, unlike the convention in machine learning over unencryped data to place the layers the other way around. The Swap is designed to be deployed in machine learning models operating on data encrypted with fully homomorphic encryption. It speeds up encrypted inference by reducing the number of expensive homomorphic multiplications by a factor of k^2 (k=AvgPool kernel size).

This research project was supported by a Stanford VPUE Major Grant for summer 2023 and used as a part of my CS Honors Thesis at Stanford University.
