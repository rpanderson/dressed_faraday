#!/usr/bin/env python

from pyzotero import zotero
import os
import shutil
from bibfix import fixbib

# chdir to folder script it in
base_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(base_folder)

# Parameters
bibfile = 'dressed_faraday.bib'
bibfix_only = False
API_KEY = '6ZeKc1MGmP2pGjZCM5lJ363C'
GROUP_ID = '1276'
COLLECTION_ID = 'DRJ6GPQG'

# Backup BibTeX file before overwriting
if os.path.exists(bibfile):
	shutil.copyfile(bibfile, bibfile + '.old')

try:
	if not bibfix_only:
		# Get all items in collection in BibTeX format
		print 'Getting items from collection...'
		zot = zotero.Zotero(GROUP_ID, 'group', API_KEY)
		zot.add_parameters(content='bibtex', order='dateAdded', limit=99)
		items = zot.collection_items(COLLECTION_ID)

		# Discard null items
		items = [x for x in items if len(x)]
		print 'Retrieved %i items from collection...' % len(items)

		# Omit language entry (in techreport, this doesn't work in revtex4-1)
		items = [x.replace('\tlanguage = {en},\n', '') for x in items]

		# Write to file
		with open(bibfile, 'wb') as f:
		    f.write('\n\n'.join(items).encode('utf-8'))
		    f.write('\n')

	# run bibfix
	print 'Running bibfix on ' + bibfile + '...'
	fixbib(bibfile)

	print 'Done!'

except Exception, err:
	print err