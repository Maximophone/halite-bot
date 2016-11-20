from distutils.core import setup, Extension

module1 = Extension('exmod',
	sources = ['exmodmodule.c'])

setup( name = 'exmod',
	version='1.0',
	description = "Test",
	author = "mfournes",
	ext_modules=[module1])