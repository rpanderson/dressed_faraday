from glob import glob
from subprocess import check_call
import shutil
from os import path
import re

mathmode = 0
old_rev = 166

new_tex = "dressed_faraday.tex"
old_tex = "dressed_faraday@%i.tex" % old_rev
diff_out = "dressed_faraday_diff.tex"

with open(diff_out, "w") as f:
    check_call(["latexdiff", "-V", "--flatten", "--math-markup=%i" % mathmode, "--mbox"
                "--packages=amsmath,hyperref", "--exclude-textcmd=section,subsection",
                "--exclude-safecmd=note,vect,Rb,abs,uvect,vls,thetaQWP,thetaN,reffig,refeq,partialD,totalD,epvec,HzWcm",
                # "--exclude-safecmd=",
                old_tex, new_tex], stdout=f, stderr=None)

# check_call(["pdflatex", diff_out,
#             "--interaction=errorstopmode", "--include-directory=.."], shell=True)
