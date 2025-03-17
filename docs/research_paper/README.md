# DynamicCompactDetect Research Paper

This directory contains the research paper and related materials for the DynamicCompactDetect (DCD) model.

## Paper Structure

The research paper is organized into the following sections:

1. **Abstract**: A brief overview of the DynamicCompactDetect model, its innovations, and key results.

2. **Introduction**: Background on object detection models, challenges in edge computing, and the motivation for developing DCD.

3. **Methodology**:
   - Architecture of DynamicCompactDetect
   - Training methodology and optimization techniques
   - Implementation details

4. **Experiments**:
   - Datasets used for training and evaluation
   - Evaluation metrics
   - Comparison with state-of-the-art models
   - Ablation studies

5. **Results**:
   - Performance benchmarks
   - Accuracy metrics
   - Inference speed analysis
   - Model size and memory footprint
   - Cold-start performance

6. **Discussion**:
   - Analysis of results
   - Limitations and future work

7. **Conclusion**:
   - Summary of contributions
   - Potential applications

8. **References**:
   - Citations to relevant literature

## Directory Contents

- `paper.md`: The main research paper document
- `/figures/`: Visualizations, charts, and diagrams used in the paper
- `/tables/`: Performance comparison tables and other tabular data
- `/data/`: Raw data used to generate figures and tables

## Generating the Paper

To generate the latest figures and data for the paper, run:

```bash
./run_dcd_pipeline.sh --paper
```

This will execute benchmarks and create updated visualizations in the appropriate directories.

Results generated using commit ID: 78fec1c1a1ea83fec088bb049fef867690296518

## Citation

If you use DynamicCompactDetect in your research, please cite our work:

```
@article{chadhar2024dynamiccompactdetect,
  title={DynamicCompactDetect: A Lightweight Object Detection Model for Edge Devices},
  author={Chadhar, Abhilash and Athya, Divya},
  year={2024},
  url={https://github.com/future-mind/dynamiccompactdetect}
}
```

## Authors

- Abhilash Chadhar
- Divya Athya

## License

The research paper and all associated content are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). 