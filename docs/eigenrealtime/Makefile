TARGETS = nonrealtime realtime realtime_check_eigen gaussian_check_eigen

all: ${TARGETS}

%:%.cpp
		g++ -ggdb $< -o $@

%_check_eigen:%.cpp
		g++ -DREALTIME_CHECKS -ggdb $< -o $@

clean:
	rm ${TARGETS}
