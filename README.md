# Twitter Sentiment Analysis

Clone the project to your local PC.

```
C:\Users\Mfonism> git clone https://github.com/mfonism/TwitterSentimentAnalysis.git
```

Create a virtual environment in the cloned project directory, activate this environment, and install dependencies into the environment.

All of this will take a while.

```
C:\Users\Mfonism> cd TwitterSentimentAnalysis
C:\Users\Mfonism\TwitterSentimentAnalysis> python -m venv env
(env) C:\Users\Mfonism\TwitterSentimentAnalysis> env/scripts/activate
(env) C:\Users\Mfonism\TwitterSentimentAnalysis> pip install -r
```

Fire up the Python interactive shell.

```
(env) C:\Users\Mfonism\TwitterSentimentAnalysis> python
```

Run the scripts.

The example below shows how to run the analysis scripts. Each of the runs will take a considerably long period to complete, and will print lots of output to the terminal, so watch out for that.

```
>>> from src.analysis import cnn
>>> cnn.run()
```

```
>>> from src.analysis import lstm
>>> lstm.run()
```
