import subprocess
import sys

# Inputs
files = ['dressed_faraday.tex']
output = 'wordcount.txt'
figureDimensions = [(9.00, 4.79), (8.94, 4.65), (9.00, 13.17), (9.00, 4.79)]
tableLines = [10]

try:
    texcount_output = subprocess.check_output('texcount -total ' + ' '.join(files))
    if not textcount_output.startswith('Total'):
        raise Exception
    else:
        with file(output, 'w') as f:
            f.write(textcount_output)
        lines = textcount_output.split('\r\n')
except:
    'Failed to run texcount... getting counts from wordcount.txt instead'
    with file(output, 'r') as f:
        lines = f.readlines()

# Word limit (PRL)
wordLimit = 3750

# Dictionary of texcount output
counts = [x.rstrip().split(': ') for x in lines if ': ' in x]
counts = dict([(x[0].split('(')[0].rstrip(), int(x[1])) for x in counts])

# Text
wordsPerInlineMath = 2
textWords = counts['Words in text'] + counts['Words outside text'] + \
    wordsPerInlineMath * counts['Number of math inlines']

# Displayed math
wordsPerDisplayMath = 16
displayMathWords = wordsPerDisplayMath * counts['Number of math displayed']

# Figures
aspectRatios = [x[0]/x[1] for x in figureDimensions]
figureWordCounts = [int(150/x) + 21 for x in aspectRatios]
figureWords = sum(figureWordCounts)

# Tables
tableWordCounts = [26 + 13 * x for x in tableLines]
tableWords = sum(tableWordCounts)

# Total
totalWords = textWords + displayMathWords + figureWords + tableWords
# print(totalWords)

# Prepare APS style table
figureTableLines = ['Figure   Aspect Ratio   Wide?   Word Equivalent']
for i, (r, w) in enumerate(zip(aspectRatios, figureWordCounts)):
    figureTableLines.append('   {:3d}         {:.2f}         No     {:6d}'.format(i+1, r, w))

tableTableLines = ['  Table   Lines   Wide?   Word Equivalent']
for i, (l, w) in enumerate(zip(tableLines, tableWordCounts)):
    tableTableLines.append('   {:3d}    {:4d}    Yes    {:6d}'.format(i+1, l, w))

summaryLines = ['WORD COUNT SUMMARY',
                '    Note: Text word count excludes title, abstract, byline, PACS,',
                '          receipt date, acknowledgments, and references.',
                '\n',
                '              Text word count {:6d}'.format(textWords),
                '        Equations word equiv. {:6d}'.format(displayMathWords),
                '          Figures word equiv. {:6d}'.format(figureWords),
                '           Tables word equiv. {:6d}'.format(tableWords),
                '                              ------',
                '                       TOTAL  {:6d} (Maximum length is {:d})'.format(totalWords, wordLimit),
                '                      EXCESS  {:6d} words'.format(totalWords - wordLimit)
                ]

allLines = figureTableLines + ['\n'] + tableTableLines + ['\n'] + summaryLines
summaryString = '\n'.join(allLines)

# Save to file and print
with file(output.replace('.txt', '_aps.txt'), 'w') as f:
    f.write(summaryString)
print(summaryString)