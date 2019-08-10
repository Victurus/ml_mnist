FILE=test1_mnist.py
PC=python3
DIR=$(shell pwd)

all: prepare run_tmux

run:
	clear
	$(PC) $(FILE)

run_tmux:
	tmux send-keys -t .+ ' clear; cd $(DIR); $(PC) $(FILE)' 'Enter'

prepare:
	[ -d './data' ] || mkdir data
	[ -d './model' ] || mkdir model

.PHONY: run run_tmux
