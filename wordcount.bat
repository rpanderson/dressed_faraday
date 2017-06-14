set output=wordcount.txt
set input=dressed_faraday.tex
start cmd.exe /k "texcount -total %input% > %output% & type %output%"