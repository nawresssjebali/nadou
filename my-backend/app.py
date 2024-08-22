from flask import Flask, request, jsonify
from flask_cors import CORS
import math
import os
import datetime

app = Flask(__name__)
CORS(app)
file_path = os.path.join(os.path.dirname(__file__), 'coord50-5-1.dat')
output_file_path = os.path.join(os.path.dirname(__file__), 'courier_data.txt')

# Read data function
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = [line.strip() for line in file.readlines()]

    num_customers = int(data[0])
    num_depots = int(data[1])
    depot_coordinates = [tuple(map(int, point.split())) for point in data[3:num_depots + 3]]
    customer_coordinates = [tuple(map(int, point.split())) for point in data[num_depots + 4:num_customers + num_depots + 4]]
    vehicle_capacity = int(data[num_customers + num_depots + 5].split()[0])
    depot_capacities = [int(data[i].split()[0]) for i in range(num_customers + num_depots + 7, num_customers + num_depots + 7 + num_depots)]
    customers_demands = [int(data[i].split()[0]) for i in range(num_customers + num_depots + 7 + num_depots + 1, num_customers + num_depots + 7 + num_depots + 1 + num_customers)]
    opening_costs_depots = [int(data[i].split()[0]) for i in range(num_customers + num_depots + 7 + num_depots + 1 + num_customers + 1, num_customers + num_depots + 7 + num_depots + 1 + num_customers + 1 + num_depots)]
    opening_cost_route = int(data[num_customers + num_depots + 7 + num_depots + 1 + num_customers + 1 + num_depots + 1].split()[0])

    return num_customers, num_depots, depot_coordinates, customer_coordinates, vehicle_capacity, depot_capacities, customers_demands, opening_costs_depots, opening_cost_route

# Helper function to calculate distance
def calculate_distance(loc1, loc2):
    return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

# Function to find customers near the courier
def find_customers_near_courier(courier_location, customer_coordinates, customers_demands, threshold_distance=5):
    nearby_customers = []
    nearby_demands = 0
    nearby_coordinates = []

    for idx, customer_location in enumerate(customer_coordinates):
        distance = calculate_distance(customer_location, courier_location)
        if distance <= threshold_distance:
            nearby_customers.append(idx)
            nearby_demands += customers_demands[idx]
            nearby_coordinates.append(customer_location)

    return nearby_customers, nearby_demands, nearby_coordinates

# Function to update data with the courier
def update_data_with_courier(file_path, courier_location, threshold_distance=5):
    num_customers, num_depots, depot_coordinates, customer_coordinates, vehicle_capacity, depot_capacities, customers_demands, opening_costs_depots, opening_cost_route = read_data(file_path)

    nearby_customers, total_demand, nearby_coordinates = find_customers_near_courier(courier_location, customer_coordinates, customers_demands, threshold_distance)

    saved_nearby_coordinates = nearby_coordinates.copy()

    customer_coordinates = [loc for idx, loc in enumerate(customer_coordinates) if idx not in nearby_customers]
    customers_demands = [demand for idx, demand in enumerate(customers_demands) if idx not in nearby_customers]

    customer_coordinates.append(courier_location)
    customers_demands.append(total_demand)

    num_customers = len(customer_coordinates)

    new_data = []
    new_data.append(str(num_customers))
    new_data.append(str(num_depots))
    new_data.extend(f"{coord[0]} {coord[1]}" for coord in depot_coordinates)
    new_data.append("")
    new_data.extend(f"{coord[0]} {coord[1]}" for coord in customer_coordinates)
    new_data.append("")
    new_data.append(str(vehicle_capacity))
    new_data.append("")
    new_data.extend(str(cap) for cap in depot_capacities)
    new_data.append("")
    new_data.extend(str(demand) for demand in customers_demands)
    new_data.append("")
    new_data.extend(str(cost) for cost in opening_costs_depots)
    new_data.append("")
    new_data.append(str(opening_cost_route))

    with open(file_path, 'w') as file:
        for line in new_data:
            file.write(f"{line}\n")

    return saved_nearby_coordinates, total_demand

@app.route('/add_courier', methods=['POST'])
def add_courier():
    data = request.json
    x = data.get('coordinateX')
    y = data.get('coordinateY')
    area_covered = data.get('area')
    price_per_kg = data.get('price')

    # Use the received data to update the file
    courier_location = (x, y)
    saved_coordinates, new_customer_quantity = update_data_with_courier(file_path, courier_location)

    # Calculate the total price
    total_price = price_per_kg * new_customer_quantity

    # Save the required information to the output file
    with open(output_file_path, 'a') as output_file:
        output_file.write(f"Date: {datetime.datetime.now()}\n")
        output_file.write(f"Courier Location: ({x}, {y})\n")
        output_file.write(f"Area Covered: {area_covered}\n")
        output_file.write("Closest Customers' Coordinates: " + ', '.join(f"({cx}, {cy})" for cx, cy in saved_coordinates) + "\n")
        output_file.write(f"New Customer Quantity: {new_customer_quantity}\n")
        output_file.write(f"Price per 1 Kg: {price_per_kg}\n")
        output_file.write(f"Total Price (1 Kg * New Customer Quantity): {total_price}\n")
        output_file.write("--------------------------------------------------\n")

    return jsonify({
        "message": "Data processed and saved successfully",
        "saved_coordinates": saved_coordinates,
        "new_customer_quantity": new_customer_quantity,
        "total_price": total_price
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
