build:

HORIZON := 1e7
SEEDCOUNT := 10
ALGORITHMS := sgd,adam,sgd-sched-1,sgd-sched-1p2,sgd-sched-2p3,stab-sgd-oracle,stab-sgd-inline,dadapt-d4,dadapt-d200,rls,cocob,cocob-backprop
PLOTTED := ${ALGORITHMS}

run:
	python -m src.run.scan_quadratic -a ${ALGORITHMS} -e=-7:5:0.5 -T ${HORIZON} -s 1:${SEEDCOUNT} -p rbs-2025 -P QSC,QWC -v 1e-16
	python -m src.run.scan_quadratic -a ${ALGORITHMS} -e=-7:5:0.5 -T ${HORIZON} -s 1:${SEEDCOUNT} -p rbs-2025 -P QWC -v 1e-4,1e0

plot:
	python -m src.plot.plot_scan -a ${PLOTTED} -P QSC,QWC -v 1e-16 -p rbs-2025 -D "export/QC" --noshow
	python -m src.plot.plot_scan -a ${PLOTTED} -P QWC -v 1e-4 -p rbs-2025 -D "export/QC" --noshow
	python -m src.plot.plot_scan -a ${PLOTTED} -P QWC -v 1 -p rbs-2025 -D "export/QC" --noshow

package:
	python -m src.plot.aggregate_scan -a ${PLOTTED} -P R-QSC,R-QWC -v 1e-16 -p rbs-2025 --tuning_horizons 1e1,1e2,1e3,1e4,1e5,1e6,1e7

install:
	pip install -r requirements.txt

clean:
	rm -rf src/.mypy_cache src/**/.mypy_cache
	rm -rf src/__pycache__
	rm -rf src/**/__pycache__
