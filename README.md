# Tiefe Neuronale Netze zur sequenziellen Klassifizierung: Sekundärstrukturvorhersage von Proteinen

## Aufbauend auf der Codebase von:
Implementation of [High Quality Prediction of Protein Q8 Secondary Structure by Diverse Neural Network Architectures](https://arxiv.org/abs/1811.07143),
[Iddo Drori](https://www.cs.columbia.edu/~idrori), [Isht Dwivedi](http://www.ishtdwivedi.in), Pranav Shrestha, Jeffrey Wan, [Yueqi Wang](https://github.com/yueqiw), Yunchu He, Anthony Mazza, Hugh Krogh-Freeman, [Dimitri Leggas](https://www.college.columbia.edu/node/11468), Kendal Sandridge, [Linyong Nan](https://github.com/linyongnan), [Kaveri Thakoor](http://www.seismolab.caltech.edu/thakoor_k.html), Chinmay Joshi, [Sonam Goenka](https://github.com/sonamg1), [Chen Keasar](https://www.cs.bgu.ac.il/~keasar), [Itsik Pe’er](http://www.cs.columbia.edu/~itsik)
NIPS Workshop on Machine Learning for Molecules and Materials, 2018.


Q3 (links) and Q8 (rechts) Sekundärstrukturen des 1AKD Protein im  CB513 Datensatz:

<img src="https://github.com/idrori/cu-ssp/blob/master/paper/figures/1akd_q3.png" height=300><img src="https://github.com/idrori/cu-ssp/blob/master/paper/figures/1akd_q8.png" height=300>

Eigene Beiträge im Ordner model_neu

Implementierungen auf Daten

https://www.princeton.edu/~jzthree/datasets/ICML2014/

http://www.cbs.dtu.dk/services/NetSurfP/

https://github.com/qzlshy/ss_pssm_hhm

## Übersicht der Ordner:
Reproduktion der Daten: /princetion
Test auf netsurfp Daten + Aufbereitung des codes: /netsurfp
Test auf qzlshy Daten: /qzlshy
Optimierte Modelle mit Hyperopt: /optimized
Vor- und Aufbereitung der Daten: /prepare_data

Die Modelle sind größtenteils für den GPU Gebrauch optimiert.

virtualenv -p python3 tf-gpu

tensorflow-gpu==1.12.0
module load cudnn/7.3.0_cuda-9.0

absl-py==0.7.0
alabaster==0.7.12
allennlp==0.8.5
asn1crypto==0.24.0
astor==0.7.1
atomicwrites==1.3.0
attrs==19.1.0
Babel==2.7.0
biotite==0.15.1
blis==0.2.4
boto==2.49.0
boto3==1.9.200
botocore==1.12.200
bson==0.5.8
certifi==2019.6.16
cffi==1.12.3
chardet==3.0.4
Click==7.0
conda==4.7.11
conda-package-handling==1.3.10
conllu==1.3.1
cryptography==2.3.1
cycler==0.10.0
cymem==2.0.2
decorator==4.4.0
dill==0.2.9
docutils==0.14
editdistance==0.5.3
flaky==3.6.1
Flask==1.1.1
Flask-Cors==3.0.8
ftfy==5.6
future==0.17.1
gast==0.2.2
gensim==3.8.0
gevent==1.4.0
greenlet==0.4.15
grpcio==1.18.0
h5py==2.9.0
hyperopt==0.1.2
idna==2.8
imagesize==1.1.0
importlib-metadata==0.23
itsdangerous==1.1.0
Jinja2==2.10.1
jmespath==0.9.4
jsonnet==0.14.0
jsonpickle==1.2
Keras==2.2.4
Keras-Applications==1.0.7
Keras-Preprocessing==1.0.8
keras-tcn==2.3.5
kiwisolver==1.1.0
libarchive-c==2.8
Markdown==3.0.1
MarkupSafe==1.1.1
matplotlib==3.1.1
more-itertools==7.2.0
msgpack==0.6.1
murmurhash==1.0.2
networkx==2.3
nltk==3.4.5
numpy==1.16.1
numpydoc==0.9.1
overrides==1.9
packaging==19.1
pandas==0.24.2
parsimonious==0.8.1
plac==0.9.6
pluggy==0.13.0
preshed==2.0.1
protobuf==3.6.1
py==1.8.0
pycosat==0.6.3
pycparser==2.19
pydot==1.4.1
Pygments==2.4.2
pymongo==3.8.0
pyOpenSSL==19.0.0
pyparsing==2.4.0
PySocks==1.7.0
pytest==5.1.2
python-dateutil==2.8.0
python-telegram-bot==12.1.1
pytorch-pretrained-bert==0.6.2
pytorch-transformers==1.1.0
pytz==2019.1
PyYAML==5.1
regex==2019.8.19
requests==2.22.0
responses==0.10.6
ruamel-yaml==0.15.46
s3transfer==0.2.1
scikit-learn==0.20.2
scipy==1.2.0
sentencepiece==0.1.83
six==1.12.0
sklearn==0.0
smart-open==1.8.4
snowballstemmer==1.9.1
spacy==2.1.8
Sphinx==2.2.0
sphinxcontrib-applehelp==1.0.1
sphinxcontrib-devhelp==1.0.1
sphinxcontrib-htmlhelp==1.0.2
sphinxcontrib-jsmath==1.0.1
sphinxcontrib-qthelp==1.0.2
sphinxcontrib-serializinghtml==1.1.3
sqlparse==0.3.0
srsly==0.1.0
style==1.1.0
tensorboard==1.12.2
tensorboardX==1.8
tensorflow==1.12.0
tensorflow-gpu==1.12.0
termcolor==1.1.0
thinc==7.0.8
torch==1.1.0
tornado==6.0.3
tqdm==4.32.1
Unidecode==1.1.1
update==0.0.1
urllib3==1.24.2
wasabi==0.2.2
wcwidth==0.1.7
Werkzeug==0.15.6
word2number==1.1
zipp==0.6.0
