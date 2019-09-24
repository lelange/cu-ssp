# Tiefe Neuronale Netze zur sequenziellen Klassifizierung: Sekundärstrukturvorhersage von Proteinen

## Aufbauend auf der Codebase von:
Implementation of [High Quality Prediction of Protein Q8 Secondary Structure by Diverse Neural Network Architectures](https://arxiv.org/abs/1811.07143),
[Iddo Drori](https://www.cs.columbia.edu/~idrori), [Isht Dwivedi](http://www.ishtdwivedi.in), Pranav Shrestha, Jeffrey Wan, [Yueqi Wang](https://github.com/yueqiw), Yunchu He, Anthony Mazza, Hugh Krogh-Freeman, [Dimitri Leggas](https://www.college.columbia.edu/node/11468), Kendal Sandridge, [Linyong Nan](https://github.com/linyongnan), [Kaveri Thakoor](http://www.seismolab.caltech.edu/thakoor_k.html), Chinmay Joshi, [Sonam Goenka](https://github.com/sonamg1), [Chen Keasar](https://www.cs.bgu.ac.il/~keasar), [Itsik Pe’er](http://www.cs.columbia.edu/~itsik)
NIPS Workshop on Machine Learning for Molecules and Materials, 2018.


Q3 (links) and Q8 (rechts) Sekundärstrukturen des 1AKD Protein im  CB513 Datensatz:

<img src="https://github.com/idrori/cu-ssp/blob/master/paper/figures/1akd_q3.png" height=300><img src="https://github.com/idrori/cu-ssp/blob/master/paper/figures/1akd_q8.png" height=300>

Trainings- und Testdaten

https://www.princeton.edu/~jzthree/datasets/ICML2014/

http://www.cbs.dtu.dk/services/NetSurfP/

https://github.com/qzlshy/ss_pssm_hhm


Gewichte für die Erstellung der ElMo Einbettung:

https://github.com/mheinzinger/SeqVec


## Übersicht der Ordner:

model_n: ursprüngliche Implementierungen der Modelle (evtl. Änderungen zur Fehlerbehebung)

Eigene Beiträge im Ordner **model_neu**

Reproduktion der Daten: /princetion

Test auf netsurfp Daten + Aufbereitung des codes: /netsurfp

Test auf qzlshy Daten: /qzlshy

Optimierte Modelle mit Hyperopt: /optimized

Vor- und Aufbereitung der Daten: /prepare_data


Die Modelle sind größtenteils für den GPU Gebrauch optimiert.


wichtige Vorausetzungen:

python3

tensorflow-gpu==1.12.0

module load cudnn/7.3.0_cuda-9.0

sonstige Paketeabhängigkeiten im Ordner: /package_dependencies
