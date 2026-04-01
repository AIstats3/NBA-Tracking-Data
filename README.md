# nba-movement-data

Ever since the nba stopped public access of their movement data, I though it would be good to have a copy of @neilmj data repo incase he deletes his data repo.

Credit: [@neilmj](https://github.com/neilmj/BasketballData)

## Data Setup

1.To unzip the `7z` files, ensure `7z` is installed on your computer first.
Then, run this command:

```shell
cd data && ./setup.sh
```

## Additional Data Conversions

1. Install the package.

(From the package root direction)

```shell
python setup.py build
pip install -e .
```

2. Convert the JSON files.

```shell
python movement/json_to_csv.py
```

3. Convert the full-court to half-court. An explanation of moving the SportVU movement can be found [here](https://github.com/sealneaward/movement-quadrants).

```shell
python movement/convert_movement.py
```

4. The fixed shot times, along with the shot locations in half court space are in `data/shots/fixed_shots.csv`. They are formed from executing the script.

```shell
python movement/fix_shot_times.py
```

In the fixing logic, the shot time is defined as the highest acceleration point before the ball reaches it's peak, within a defined window.
The logic can be seen below.
![plot](movement/plot.png)
