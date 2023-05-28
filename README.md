# Thesis

A Deep Learning Architecture for Enhanced Depth Estimation on an unrectified Stereo 360◦ dataset<img width="1050" alt="image" src="https://github.com/danialzendehdel/Thesis/assets/49899830/3b0ba387-5375-48ab-a1bc-a27b746fce3c">

Stereo depth estimation is a critical problem in computer vision, with applications in robotics, self-driving vehicles, unmanned aerial vehicles, virtual and augmented reality, and 3D model reconstruction. While deep learning has led to significant improvements in depth estimation, existing datasets are predominantly synthetic and do not fully represent the complexity of real-world scenes or 360$^\circ$ images. This thesis presents a novel dataset for depth estimation from stereo 360$^\circ$ images and explores the challenges and opportunities it offers.

The dataset has several unique characteristics: it includes real-world images, features a vertical camera and LiDAR setup, contains unrectified image pairs, and provides depth images as ground-truth instead of disparity maps. These properties make the dataset more challenging and better suited for real-world applications.

We evaluate the performance of popular depth estimation architectures, such as PSMNet and 360SD-Net, on our dataset and identify their strengths and weaknesses in handling stereo 360$^\circ$ images. To further improve performance, we modify and add code to these architectures, demonstrating that our dataset presents a new challenge for deep learning tasks in stereo-depth estimation. By achieving satisfactory results at the primary level, we provide a foundation for future research in this area.

The thesis is organized into six chapters: Introduction, State of the Art, Proposed Dataset, Deep Learning Architectures, Experiments and Results, and Conclusion. The work presented in this thesis contributes to advancing the field of stereo depth estimation and paves the way for further 
