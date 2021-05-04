# Vorlage für die Abschlussarbeit

Diese Sammlung an tex-Dateien soll als Vorlage und Ausganspunkt für das
Erstellen einer Abschlussarbeit mit Hilfe von LuaLaTeX dienen. Die Vorlage
müsste aber auch mit pdfLaTeX funktionieren.

Das ganze ist modular aufgebaut: für jedes Kapitel soll eine separate
Datei im Verzeichnis *chapters/* erstellt werden, diese werden dann in der
Datei *Thesis.tex* mit `\include{chapters/…}` eingelesen.
Das deklarieren der Pakete erfolgt in der Datei *preamble.tex* in der
auch alle Einstellungen vorgenommen werden.

Erstellt wird die pdf-Datei über `lualatex Thesis` oder mit einem
geeigneten Editor, der das direkte Erstellen ermöglicht (wobei hier auch
LuaLaTeX zur Erzeugung des PDFs ausgewählt werden muss).
Hier sei TeXstudio empfohlen, vor allem wegen der
Möglichkeit leicht zwischen Quelltext und entsprechender Stelle im fertigen
Dokument hin- und herzuspringen.

Noch einfacher wird es mit `latexmk -lualatex Thesis`, dieses Programm
kümmert sich automatisch darum auch externe Programme wie `biber` für
die Bibliographie aufzurufen und auch das mehrmalige Durchlaufen von
LuaLaTeX. Das kann meistens auch im verwendeten Editor konfiguriert werden.

## Hinweise
Die englische Variante ist ebenso wie die Variante mit `pdflatex` (noch)
nicht vollständig umgesetzt. Für erstere gibt es aber zumindest Bestrebungen
dies in Angriff zu nehmen.

Das Paket *polyglossia* muss mindest die Version 1.43 haben.

## Urheberrecht und so

Die Vorlage darf frei verwendet werden, insbesondere denke ich, dass sie
sowieso nicht wirklich die Schöpfungshöhe erreicht. Allerdings würde ich
mich trotzdem freuen, wenn die Hinweise auf den Urheber (im Quelltext)
erhalten bleiben würden.

ⓒ (CC0) 2016, 2018 Henning Iseke <h_i_@online.de>
