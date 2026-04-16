# Machine-Learning-Project
Final project for Introduction to Machine Learning:
3. Pneumonia Detection using CNN in Python

Hanna: Data Architect (Input & Preprocessing)

- Focuses on the Input Layer, managing the transition from raw image files to a matrix of pixels that the computer can recognize
- Dataset Management: Download and organize the RSNA Pneumonia Detection Challenge dataset, which contains thousands of X-ray images
- Structure: Programmatically divide the data into training, testing, and validation sets, ensuring each is further categorized into "pneumonia" and "normal" folders
Image Standardization: Write a preprocessing script to resize all images to 224x224x3 (RGB) as required by the VGG16 input specifications 
- GitHub Contribution: Create a data_pipeline.py script that the others can call to feed images into the model.

Lilla: Model Engineer (VGG16 Architecture)

- Responsible for the Feature Learning phase, building the complex VGG16 structure using the Keras library for optimization
- Convolutional Blocks: Implement the 13 convolutional layers using specific filters: two layers of 64, two layers of 128 (referenced as 124 in some texts), three layers of 256, and six layers of 512
- Downsampling: Integrate max-pooling layers after each block to reduce the number of parameters and save memory 7. They must use a stride of 2 for the first three blocks and a stride of 1 for the final block as specified
- Customization: Ensure the filters use the required 3x3 kernel size to simplify the stacking process 
- GitHub Contribution: Create a model_factory.py script that defines the network architecture.

Dori: Training & Analytics Lead (Optimization & Evaluation)

- Manages the Classification and refinement stage, ensuring the system reaches the high accuracy required for medical diagnostics.
- Dense Layers: Implement the fully connected hidden layer (4096 units) and the final SoftMax classifier to obtain class scores.
- Optimization Logic: Set up the training loop where the model calculates errors and backtracks to update the weights of the filters until the error is minimized.
- Performance Metrics: Derive a confusion matrix to analyze how accurately the model classifies benign vs. malignant (or in this case, pneumonia vs. normal) instances.
- GitHub Contribution: Create a main_trainer.py script that imports the data and model scripts to execute the training and generate reports.