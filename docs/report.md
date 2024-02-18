# Sprawozdanie
Mikołaj Garbowski

Przedmiotem zadania jest implementacja generatora danych opartego o sieć bayesowską 
oraz zbadanie działania zaimplementowanego wcześniej klasyfikatora opartego o algorytm ID3.

## Generator danych
Program `main.py` przyjmuje rozmiar zbioru danych i ścieżkę do pliku reprezentującego sieć,
wypisuje na wyjście standardowe dane zgodnie z formatem .csv

```
Chair,Sport,Back,Ache
1,0,0,0
1,0,1,1
1,0,0,0
1,0,0,0
1,0,0,0
1,0,0,0
1,0,0,0
1,0,0,0
0,0,0,0
1,0,0,0
```

Jak widać uruchamiając skrypt `test.py` liczba rekordów w każdej klasie jest zbliżona do wartości oczekiwanych,
wynikających z zadanego rozkładu (wartości uśrednione z 25 uruchomień)

* Chair: 81.32 (oczekiwane 80)
* Sport: 1.76 (oczekiwane 2) 
* Back: 17.32 (oczekiwane 17.676)
* Ache: 20.12 (oczekiwane 20.61)


## Wyniki klasyfikacji
Wyniki klasyfikacji modelem drzewa decyzyjnego budowanego zgodnie z algorytmem ID3.
Zbiór danych o rozmiarze 1000 wygenerowany opisanym wcześniej programem.

Wyniki są stosunkowo dobre, model daje wysoką dokładność na poziomie 85%.
Spostrzeżenie: w każdym wierszu tabeli rozkładu stosunki prawdopodobieństw są duże (np. 0,9:0,1; 0,8:0,2).
W danych wygenerowanych na podstawie takiej sieci da się sensownie wyznaczyć decyzje w drzewie.
Z drugiej strony, przy zadanym rozkładzie, dokładność 80% osiągnąłby model, który zawsze przewiduje tą samą klasę.

Za wartość pozytywną przyjmuję ból pleców (1), a za negatywną brak bólu pleców (0).

### Dla sieci z polecenia
```
Average values over 25 runs on Back pain 1 dataset
Number of samples in test set: 400
Accuracy:    84.80%
Precision:   68.93%
Recall:      56.70%
Specificity: 92.74%

TP=50     FN=38    
FP=23     TN=289  
```

### Dla zmodyfikowanej sieci
Wyniki klasyfikacji dla sieci o zmodyfikowanym rozkładzie Back, tak, że wszystkie 
prawdopodobieństwa w tabeli wynoszą 0,5.

Wyniki są słabsze niż dla poprzedniego zbioru, jednak dalej stosunkowo dobre z dokładnością 78%.
Czułość modelu jest znacznie lepsza dla tego zbioru danych.

Dla bardziej równomiernego rozkładu trudniej jest wyznaczyć dobre decyzje w drzewie, granice między klasami 
są mniej wyraźne.

```
Average values over 25 runs on Back pain 2 dataset
Number of samples in test set: 400
Accuracy:    78.15%
Precision:   67.65%
Recall:      84.77%
Specificity: 73.94%

TP=133    FN=24    
FP=64     TN=180 
```
