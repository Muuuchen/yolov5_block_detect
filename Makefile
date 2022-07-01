CLASSDIR= .
ifdef ip

else
	ip=None
endif
ifdef port

else
	port=None
endif

default: run

run:server.py
	@echo "Running server..."
	python3 ${CLASSDIR}/server.py --ip=${ip} --port=${port}

clean:
	@echo "cleaning"
	rm -rf ./img/*.jpg