In this practical session, I implemented several RNN models to perform different tasks. To be more specific:

- In `tp5-tagging.py` file, I implemented a tagging model focused on the task of syntactic analysis (Part-Of-Speech) using an LSTM. It was trained on the [GSD dataset](https://github.com/UniversalDependencies/UD_French-GSD).
- In `tp5-traduction.py` file, I implemented a translation model using a GRU. It was trained on the dataset contained in the `en-fra.txt` file.
- The `segmentation.py` file contains two `sentencepiece` segmentation model for the English and French datasets in `en-fra.txt`.

For the subject of this practical session, please refer to the [website of the course](https://dac.lip6.fr/master/amal-2024-2025/) for more details.
