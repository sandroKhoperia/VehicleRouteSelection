import pandas as pd
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm
# function to calculate the distance between two points using the Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 3959  # Earth radius in kilometers

    # convert latitude and longitude to radians
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    # apply Haversine formula
    a = sin(d_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(d_lon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c

def create_node_mapping(cities):
    node_map = {index: city for index, city in enumerate(cities.values)}
    reverse_node_map = dict((v, k) for k, v in node_map.items())
    node_map.update(reverse_node_map)

    return node_map



def compute_truck_and_cars(n_cars, n_trucks, gas_per_mile):
    total_cities = n_cars+n_trucks
    # load the data from uscities.csv
    df = pd.read_csv('uscities.csv', usecols= ['city', 'lat', 'lng'], nrows=total_cities)
    df_nodes = pd.read_csv('Dataset.csv', nrows=n_cars, usecols=['Price Per Mile','Weight'], thousands=',')
    df_nodes['Weight'] = df_nodes['Weight'].astype('int16')
    df_nodes['Price Per Mile'] = df_nodes['Price Per Mile'].astype('float16')
    print(df_nodes.dtypes)
    # extract the latitude, longitude, and city name of the first n rows
    latitudes = df['lat']
    longitudes = df['lng']
    city_names = df['city']
    weights = df_nodes['Weight']
    prices = df_nodes['Price Per Mile']
    mapping = create_node_mapping(city_names)
    print(len(latitudes), len(longitudes), len(city_names), len(weights), len(prices))
    cars = pd.DataFrame(data={
        'id': [],
        'dest_node_id': [],
        'dest_profit': [],
        'car_weight': []
    })
    trucks = pd.DataFrame(data={
        'id': [],
        'node_id': [],
        'visit_cost': [],
    })
    #new_data ={'id': 1,'dest_node_id': 2,'dest_profit': 1,'weight': 4}
    #new_node = pd.DataFrame([new_data])
    #nodes = pd.concat([nodes, new_node], ignore_index=True)
    distance_matrix = [[0] * (n_cars) for i in range(n_cars)]
    hash_set = set()
    for i in tqdm(range(total_cities), desc="Preparing dataset"):
        for j in range(n_cars):
            if i != j:
                #distance between nodes i and j
                distance = haversine(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
                #distance cost to travel from node i to node j
                distance_cost = distance*gas_per_mile

                if i >= n_cars:
                    new_data = {'id': i,'type': 1, 'node_id': j, 'visit_cost': distance_cost }
                    trucks = pd.concat([trucks, pd.DataFrame([new_data])], ignore_index=True)

                else:
                    distance_matrix[i][j] = distance_cost

                if j < n_cars and j not in hash_set:
                    delivery_profit = distance*prices[j]
                    new_data = {'id': j, 'type': 0,'dest_node_id': j-3 if j >= 3 else n_cars-1-j,
                                'dest_profit': delivery_profit, 'car_weight': weights[j] }
                    cars = pd.concat([cars, pd.DataFrame([new_data])], ignore_index=True)
                    hash_set.add(j)

    trucks.to_csv('trucks.csv', index=False, header=True)
    cars.to_csv('cars.csv', index=False, header=True)
    pd.DataFrame(distance_matrix).to_csv('adj_list.csv', index=False, header=False)
    return trucks, cars, distance_matrix, create_node_mapping(city_names)

def compute_matrices(number_of_cities, gas_per_mile):
    # load the data from uscities.csv
    df = pd.read_csv('uscities.csv', nrows=number_of_cities)


    # extract the latitude, longitude, and city name of the first n rows
    latitudes = df.iloc[:number_of_cities]['lat']
    longitudes = df.iloc[:number_of_cities]['lng']
    city_names = df.iloc[:number_of_cities]['city']
    mapping = create_node_mapping(city_names)
    print(mapping)
    # copy the first city name and insert it at the beginning of city_names
    first_city = city_names[0]
    city_names = pd.concat([pd.Series([first_city]), city_names])


    # create an empty matrix to store the distances
    distance_matrix = [[0] * (number_of_cities + 1) for i in range(number_of_cities + 1)]
    gas_matrix = [[0] * (number_of_cities + 1) for i in range(number_of_cities + 1)]

    # fill in the distance matrix with progress bar
    for i in tqdm(range(1, number_of_cities + 1), desc="Calculating distance matrix"):
        # insert the city name in the first column
        distance_matrix[i][0] = city_names.iloc[i]
        distance_matrix[0][i] = city_names.iloc[i]
        gas_matrix[i][0] = city_names.iloc[i]
        gas_matrix[0][i] = city_names.iloc[i]

        for j in range(1, number_of_cities + 1):
            # fill in the distance matrix with distances
            distance_matrix[i][j] = haversine(latitudes[i - 1], longitudes[i - 1], latitudes[j - 1], longitudes[j - 1])
            gas_matrix[i][j] = distance_matrix[i][j] * gas_per_mile

    # create a DataFrame to store the distance matrix
    df_distance_matrix = pd.DataFrame(distance_matrix)
    df_gas_matrix = pd.DataFrame(gas_matrix)

    # set the column names of the distance matrix to the city names
    df_distance_matrix.set_axis(city_names, axis=0, copy=True)
    df_distance_matrix.set_axis(city_names, axis=1, copy=True)

    df_gas_matrix.set_axis(city_names, axis=0, copy=True)
    df_gas_matrix.set_axis(city_names, axis=1, copy=True)

    return df_distance_matrix, df_gas_matrix

def save_files(distance_matrix, gas_matrix):
    # save the result to a file
    distance_matrix.to_csv('distance_matrix.csv', index=False, header=False)
    gas_matrix.to_csv('gas_matrix.csv', index=False, header=False)


def main():
    gas_per_mile = 0.15
    number_of_cities = 30
    compute_truck_and_cars(number_of_cities, 5, gas_per_mile)

    #d_matrix, g_matrix = compute_matrices(number_of_cities, gas_per_mile)
    #save_files(d_matrix, g_matrix)



if __name__ == '__main__':
    main()