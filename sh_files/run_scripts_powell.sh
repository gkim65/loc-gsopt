# python src/main.py --multirun problem.gs_num=1,2,3,4,5,7,10,15,20 debug.randseed=0,1,2,3,4

# python src/main.py --multirun scenario.start_simplex=0,1,2,3,4,5,6

for gs_num in 1 2 3 4 5 7; do
  for randseed in 5 6 7 8 9 10; do
    python src/main.py problem.gs_num=$gs_num debug.randseed=$randseed problem.method="powell"
    sleep 3
  done
done

echo "All scripts executed."
