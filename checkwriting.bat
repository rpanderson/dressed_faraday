set output=dressed_faraday.html
set input=dressed_faraday.tex
start cmd.exe /c "perl %USERPROFILE%/.bin/checkwriting/checkwriting %input% | ansi2html > %output%"