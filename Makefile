CLASSDIR= .
ifdef ip

else
	ip=None
endif
ifdef port

else
	port=None
endif

default: S

S:${CLASSDIR}/classes/server.py
	@echo "Running server..."
	python3 ${CLASSDIR}/classes/server.py --ip=${ip} --port=${port}

C:${CLASSDIR}/detect.py
	@echo "Running detect..."
	python3 ${CLASSDIR}/detect.py

clean:
	@echo "cleaning"
	rm -rf ./img/*.jpg