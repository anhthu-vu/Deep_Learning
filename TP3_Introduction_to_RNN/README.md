In this practical session, I implemented a simple RNN from sratch and then used it to build and train different models for various tasks: classification, forcasting, and text generation. To be more specific:

- In utils.py file, I implemented a simple RNN with one layer for the encoder and one layer for the decoder.
- In exo2.py file, I used the RNN to build and train a classification model that takes an input sequence of the flows of metro stations in Hangzhou, and predicts which station the sequence belongs to.
- In exo3.py file, I used the RNN to build and train a forecasting model that takes an input sequence of the flows of metro stations in Hangzhou for t successive quarter-hour slices, and predict the flows at time t+1.
- In exo4.py file, I used the RNN to build and train a text generation model that takes an input sequence of symbols (letters, punctuations, digits and space), and produces the next symbols of the sequence. This model is restricted to the generation of fixed-size sequences that will be determined in advance.

For the subject of this practical session, please refer to the website of the course: https://dac.lip6.fr/master/amal-2024-2025/ for more details.
