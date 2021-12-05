# Attention is All you need

## Dataset
multi30k

## 1. Pre requirements
```python
pip install torchtext
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## 2. Train the model
```python
python main.py --mode=train --epochs=10 --lr=0.0005 --batch_size=258
```

## 3. Calculate BLEU score with the trained model
```python
python main.py --mode=test
```