# Bachelor Thesis
## **Erstellung eines Systems zur Volatilitätsprediction von Finanzmarktdaten**
### Thesis von Tobias Vincent Esser
#### Abstract
Bestimmte Trader legen bei ihren Transaktionen sehr viel Wert auf Risikominimierung. Um zu bestimmen, was für ein Risiko bei einem Kauf eingegangen wird, greifen sie insbesondere auf Daten zur Volatilität der Aktien zurück. Besonders informativ ist dabei die prognostizierte zukünftige Volatilität der Aktien.<br>

Die vorliegende Arbeit behandelt die Volatiltätsprediktion mit Hilfe von rekurrenten neuronalen Netzen. Mithilfe von Modellen sollen Volatilitäten in Finanzmarktdaten prognostiziert werden. Dazu wurde folgende Forschungsfrage entwickelt: Wie und mit welcher Genauigkeit sind rekurrente neuronale Netze in der Lage, die Volatilität von Finanzmarktdaten voraus zu sagen? Außerdem wurde der Ansatz verfolgt, unterschiedliche rekurrente neuronale Netze im Bezug auf das Prognosepotential der jeweiligen Arten zu vergleichen.<br>

Zu diesem Zweck wurden vier verschiedene Modelle erstellt und mit Daten des deutschen Aktienindex DAX trainiert. Auf Basis des Trainings prognostizierten die Modelle eine gewisse Anzahl an Volatilitätswerten. Mithilfe dieser Vorhersagewerte wurden die Modelle dann auf unterschiedliche Aspekte analysiert und interpretiert.<br>

Die Ergebnisse zeigen, dass komplexere Modelle wie ein Long-Short-Term-Memory (LSTM), ein Gated Recurrent Unit (GRU) oder Kombinationen aus LSTM und GRU bessere Ergebnisse liefern als simplere rekurrente neuronale Netze, da erstere mehr Informationen aus den vorangegangenen Daten nutzen können und so eine Art Langzeitgedächtnis entwickeln. <br>

Weitere Forschung könnte Bezug auf andere Daten nehmen und so ein General-Purpose-Modell für unterschiedliche Indizes erstellen.
#### Code
Dieses Repository beinhaltet den Code zur Bachelor Thesis von **Tobias Vincent Esser**.
Es besteht aus 5 Python-Skripten:
* Function-Collection
  * [Utils](utils.py)
* Modelle
  * [SimpleRNN](simplernn.ipynb)
  * [Long-Short-Term-Memory (LSTM)](lstm.ipynb)
  * [Gated Recurrent Unit (GRU)](gru.ipynb)
  * [Erweitertes RNN](extended_rnn.ipynb)
