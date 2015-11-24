FANNDIR=/home/gaoxiang/fann-build/src
CC=gcc
CFLAGS=-I $(FANNDIR)/include -L$(FANNDIR) -static -lfann -lm -O -std=gnu11

.PHONY:all clean

TGTS=test ann ann-cv ann-cv-l1sigm stat-train
all:$(TGTS)

%:%.c
	$(CC) $< $(CFLAGS) -o $@

clean:
	rm -rf $(TGTS)
