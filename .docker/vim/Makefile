#!/bin/bash

vim: /workspace/.vim/autoload/pathogen.vim ~/.vim/bundle/vim-go ~/.vim/pack/vendor/start/VimCompletesMe

/workspace/.vim/autoload/pathogen.vim:
	mkdir -p ~/.vim/autoload ~/.vim/bundle && \
	curl -LSso ~/.vim/autoload/pathogen.vim https://tpo.pe/pathogen.vim

~/.vim/bundle/vim-go:
	git clone https://github.com/fatih/vim-go.git ~/.vim/bundle/vim-go

~/.vim/pack/vendor/start/VimCompletesMe:
	mkdir -p  ~/.vim/pack/vendor/start
	git clone git://github.com/ajh17/VimCompletesMe.git ~/.vim/pack/vendor/start/VimCompletesMe

