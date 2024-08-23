from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import math
import os
import datetime
import math
import matplotlib.pyplot as plt
import random
import copy
import numpy as np
from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxIterations
from sklearn.cluster import KMeans
import matplotlib.patches as patches
from flask_cors import CORS  # Import CORS
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})





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
def read_data_1(file_path):
    with open(file_path, 'r') as file:
        data = [line.strip() for line in file.readlines()]

    num_customers = int(data[0])  # Line 1: Number of customers
    num_depots = int(data[1])  # Line 2: Number of depots

    # Lines 3 to 7: Depot coordinates
    depot_coordinates = [tuple(map(int, point.split())) for point in data[2:2 + num_depots]]

    # Lines 9 to 58: Customer coordinates
    customer_coordinates = [tuple(map(int, point.split())) for point in data[3 + num_depots:3 + num_depots + num_customers]]

    # Line 60: Vehicle capacity
    vehicle_capacity = int(data[4 + num_depots + num_customers].split()[0])

    # Lines 62 to 66: Depot capacities
    depot_capacities = [int(data[i].split()[0]) for i in range(6 + num_depots + num_customers, 6 + num_depots + num_customers + num_depots)]

    # Lines 68 to 117: Customer demands
    customers_demands = [int(data[i].split()[0]) for i in range(7 + num_depots + num_customers + num_depots, 7 + num_depots + num_customers + num_depots + num_customers)]

    # Lines 119 to 123: Depot opening costs
    opening_costs_depots = [int(data[i].split()[0]) for i in range(8 + num_depots + num_customers + num_depots + num_customers, 8 + num_depots + num_customers + num_depots + num_customers + num_depots)]

    # Line 125: Route opening cost
    opening_cost_route = int(data[-1].split()[0])

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
def update_data_with_courier_1(file_path, courier_location, threshold_distance):
    num_customers, num_depots, depot_coordinates, customer_coordinates, vehicle_capacity, depot_capacities, customers_demands, opening_costs_depots, opening_cost_route = read_data_1(file_path)

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

# Function to update data with the courier
def update_data_with_courier(file_path, courier_location, threshold_distance):
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
    
    # Extract the data sent from the frontend
    x = data.get('coordinateX')
    y = data.get('coordinateY')
    area_covered = data.get('area')
    price_per_kg = data.get('price')
    identifier = data.get('identifier')  # Get the identifier of the Add button
    
    # Log the identifier
    print(f"Identifier received: {identifier}")

    # Use the appropriate function based on the identifier
    courier_location = (x, y)
    
    if identifier == "add-btn-0":
        saved_coordinates, new_customer_quantity = update_data_with_courier(file_path, courier_location, area_covered)
    else:
        saved_coordinates, new_customer_quantity = update_data_with_courier_1(file_path, courier_location, area_covered)

    # Calculate the total price
    total_price = price_per_kg * new_customer_quantity

    # Save the required information to the output file
    with open(output_file_path, 'a') as output_file:
        output_file.write(f"Date: {datetime.datetime.now()}\n")
        output_file.write(f"Identifier: {identifier}\n")
        output_file.write(f"Courier Location: ({x}, {y})\n")
        output_file.write(f"Area Covered: {area_covered}\n")
        output_file.write("Closest Customers' Coordinates: " + ', '.join(f"({cx}, {cy})" for cx, cy in saved_coordinates) + "\n")
        output_file.write(f"New Customer Quantity: {new_customer_quantity}\n")
        output_file.write(f"Price per 1 Kg: {price_per_kg}\n")
        output_file.write(f"Total Price (1 Kg * New Customer Quantity): {total_price}\n")
        output_file.write("--------------------------------------------------\n")

    return jsonify({
        "message": "Data processed and saved successfully",
        "identifier": identifier,
        "saved_coordinates": saved_coordinates,
        "new_customer_quantity": new_customer_quantity,
        "total_price": total_price
    }), 200
import math
import matplotlib.pyplot as plt
import random
import copy
import numpy as np
from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxIterations
from sklearn.cluster import KMeans

# Number of commodities
num_commodities = 1

# Reading data from file

def create_potential_hub_locations(depot_coordinates, opening_costs_depots, depot_capacities):
    nodes = []
    for idx, (location, cost) in enumerate(zip(depot_coordinates, opening_costs_depots)):
        initial_quantities = [depot_capacities[idx] // num_commodities] * num_commodities
        nodes.append({
            'id': idx,
            'location': location,
            'type': 'hub',
            'cost': cost,
            'capacity': depot_capacities[idx],
            'initial_quantities': initial_quantities
        })
    return nodes

import math
import matplotlib.pyplot as plt
import random
import copy
import numpy as np
from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxIterations
from sklearn.cluster import KMeans

# Number of commodities
num_commodities = 1

# Reading data from file
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

# Helper functions
def create_potential_hub_locations(depot_coordinates, opening_costs_depots, depot_capacities):
    nodes = []
    for idx, (location, cost) in enumerate(zip(depot_coordinates, opening_costs_depots)):
        initial_quantities = [depot_capacities[idx] // num_commodities] * num_commodities
        nodes.append({
            'id': idx,
            'location': location,
            'type': 'hub',
            'cost': cost,
            'capacity': depot_capacities[idx],
            'initial_quantities': initial_quantities
        })
    return nodes

def create_nodes(customers_demands, customer_coordinates):
    nodes = []
    for idx, location in enumerate(customer_coordinates):
        demand = customers_demands[idx]
        commodity = idx % num_commodities  # Distribute commodities round-robin
        nodes.append({
            'id': idx,
            'location': location,
            'type': 'customer',
            'demand': demand,
            'commodity': commodity,
            'initial_quantities': [0] * num_commodities
        })
    return nodes

def create_courier_nodes(num_couriers, max_demand, max_cost, grid_size):
    step_size = grid_size // 3  # divide the grid into a 3x2 grid
    centers = [
        (step_size // 2, step_size // 2),
        (step_size + step_size // 2, step_size // 2),
        (2 * step_size + step_size // 2, step_size // 2),
        (step_size // 2, step_size + step_size // 2),
        (step_size + step_size // 2, step_size + step_size // 2),
        (2 * step_size + step_size // 2, step_size + step_size // 2)
    ]
    return [{'id': idx, 'demand': random.randint(1, max_demand), 'location': center, 'type': 'courier', 'cost_per_unit': random.uniform(0, max_cost), 'commodity': random.randint(0, num_commodities-1)} for idx, center in enumerate(centers)]

def calculate_distance(loc1, loc2):
    return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

# Print initial quantities at each node
def print_initial_quantities(nodes):
    print("Initial Quantities at Each Node:")
    for node in nodes:
        if node['type'] == 'customer':
            print(f"Customer Node ID: {node['id']}, Location: {node['location']}, Initial Quantities: {node['initial_quantities']}, Demand: {node['demand']}, Commodity: {node['commodity']}")
        elif node['type'] == 'hub':
            print(f"Hub Node ID: {node['id']}, Location: {node['location']}, Initial Quantities: {node['initial_quantities']}")
    print("\n")

def initialize_solution(nodes, potential_hub_locations, depot_capacities, vehicle_capacity, opening_cost_route, distance_threshold=10):
    best_solution = None
    best_cost = float('inf')

    for _ in range(100):  # Try multiple different initializations and pick the best one
        solution = _initialize_kmeans_solution(nodes, potential_hub_locations, depot_capacities, vehicle_capacity, distance_threshold)
        if solution is not None:
            # Ensure that vehicle_capacity and opening_cost_route are passed
            cost = LrpState(solution, vehicle_capacity, opening_cost_route).evaluate_solution()
            if cost < best_cost:
                best_solution = solution
                best_cost = cost

    # Handle unassigned nodes by assigning them to couriers
    if best_solution:
        handle_unassigned_with_couriers(best_solution, nodes, vehicle_capacity, distance_threshold)

    return best_solution


def _initialize_kmeans_solution(nodes, potential_hub_locations, depot_capacities, vehicle_capacity, distance_threshold):
    initial_num_hubs = random.randint(1, len(potential_hub_locations))
    initial_hubs = random.sample(potential_hub_locations, k=initial_num_hubs)
    initial_hub_nodes = [{'id': loc['id'], 'location': loc['location'], 'cost': loc['cost'], 'capacity': depot_capacities[loc['id']], 'opened': 1, 'commodity': None} for loc in initial_hubs]
    solution = {
        'num_hubs': initial_num_hubs,
        'hubs': initial_hub_nodes,
        'allocations': {hub['id']: [] for hub in initial_hub_nodes},
        'routes': {hub['id']: [] for hub in initial_hub_nodes},
        'hub_loads': {hub['id']: {commodity: 0 for commodity in range(num_commodities)} for hub in initial_hub_nodes},
        'courier_deliveries': {hub['id']: {commodity: [] for commodity in range(num_commodities)} for hub in initial_hub_nodes}
    }

    for hub in initial_hubs:
        if sum(hub['initial_quantities']) != hub['capacity']:
            print(f"Hub {hub['id']} has initial quantities {hub['initial_quantities']} that do not sum up to its capacity {hub['capacity']}")
            return None

    remaining_customers = [node for node in nodes if node['type'] == 'customer']
    assigned_customers = set()

    if len(remaining_customers) > 0 and initial_num_hubs > 1:
        kmeans = KMeans(n_clusters=min(initial_num_hubs, len(remaining_customers)))
        node_locations = np.array([node['location'] for node in remaining_customers])
        kmeans.fit(node_locations)
        labels = kmeans.labels_

        label_to_hub_id = {label: hub['id'] for label, hub in enumerate(initial_hub_nodes)}

        for idx, node in enumerate(remaining_customers):
            hub_id = label_to_hub_id[labels[idx]]
            solution['allocations'][hub_id].append(node)
            solution['hub_loads'][hub_id][node['commodity']] += node['demand']
            assigned_customers.add(node['id'])

        feasible = all(sum(solution['hub_loads'][hub['id']].values()) <= hub['capacity'] for hub in initial_hub_nodes)
    else:
        feasible = True
        for node in remaining_customers:
            if node['id'] in assigned_customers:
                continue

            closest_hub = min((hub for hub in initial_hub_nodes if sum(solution['hub_loads'][hub['id']].values()) + node['demand'] <= hub['capacity']),
                              key=lambda hub: calculate_distance(node['location'], hub['location']),
                              default=None)
            if closest_hub:
                hub_id = closest_hub['id']
                solution['allocations'][hub_id].append(node)
                solution['hub_loads'][hub_id][node['commodity']] += node['demand']
                assigned_customers.add(node['id'])
            else:
                feasible = False
                print(f"Failed to assign node {node['id']} with demand {node['demand']} and commodity {node['commodity']} to any hub.")
                break

    if feasible:
        for hub in solution['hubs']:
            hub_id = hub['id']
            depot_location = hub['location']
            cluster_nodes = solution['allocations'][hub_id]
            routes = generate_routes_with_couriers(cluster_nodes, vehicle_capacity, solution['courier_deliveries'][hub_id], depot_location)
            solution['routes'][hub_id] = routes

        non_empty_hubs = [hub for hub in solution['hubs'] if any(solution['hub_loads'][hub['id']][commodity] > 0 for commodity in range(num_commodities))]
        solution['hubs'] = non_empty_hubs
        solution['allocations'] = {hub['id']: solution['allocations'][hub['id']] for hub in non_empty_hubs}
        solution['routes'] = {hub['id']: solution['routes'][hub['id']] for hub in non_empty_hubs}
        solution['hub_loads'] = {hub['id']: solution['hub_loads'][hub['id']] for hub in non_empty_hubs}

        return solution
    else:
        print("Initialization failed: Solution is not feasible.")
        return None

def handle_unassigned_with_couriers(solution, nodes, vehicle_capacity, distance_threshold=10):
    unassigned_nodes = [node for node in nodes if node['id'] in solution.get('unassigned_nodes', [])]
    for node in unassigned_nodes:
        closest_hub_id = None
        closest_distance = float('inf')

        for hub in solution['hubs']:
            hub_id = hub['id']
            if sum(solution['hub_loads'][hub_id][node['commodity']] + node['demand'] <= hub['capacity']):
                distance = calculate_distance(hub['location'], node['location'])
                if distance < closest_distance:
                    closest_distance = distance
                    closest_hub_id = hub_id

        if closest_hub_id is not None:
            assign_to_courier(node, solution, closest_hub_id, vehicle_capacity, distance_threshold)
        else:
            print(f"No suitable hub found for node {node['id']} even with couriers.")

def assign_to_courier(node, solution, hub_id, vehicle_capacity, distance_threshold=10):
    hub_location = [hub['location'] for hub in solution['hubs'] if hub['id'] == hub_id][0]
    closest_courier = None
    min_distance = float('inf')

    for courier, customer in solution['courier_deliveries'][hub_id][node['commodity']]:
        distance = calculate_distance(courier['location'], node['location'])
        if distance < min_distance:
            min_distance = distance
            closest_courier = courier

    if closest_courier and min_distance < distance_threshold:
        solution['courier_deliveries'][hub_id][node['commodity']].append((closest_courier, node))
        solution['unassigned_nodes'].remove(node)
        solution['allocations'][hub_id].append(node)
        solution['hub_loads'][hub_id][node['commodity']] += node['demand']
        print(f"Assigned node {node['id']} to courier {closest_courier['id']} in hub {hub_id}.")
    else:
        print(f"Unable to assign node {node['id']} to any courier, even after fallback.")

class LrpState:
    def __init__(self, solution, vehicle_capacity, opening_cost_route):
        self.routes = solution['routes']
        self.hubs = solution['hubs']
        self.allocations = solution['allocations']
        self.hub_loads = solution['hub_loads']
        self.unassigned_nodes = []
        self.vehicle_capacity = vehicle_capacity  # Store vehicle capacity
        self.courier_deliveries = solution.get('courier_deliveries', {})  # Initialize courier_deliveries
        self.opening_cost_route = opening_cost_route  # Initialize opening cost for routes

    def copy(self):
        new_state = LrpState({
            'routes': copy.deepcopy(self.routes),
            'hubs': copy.deepcopy(self.hubs),
            'allocations': copy.deepcopy(self.allocations),
            'hub_loads': copy.deepcopy(self.hub_loads),
            'courier_deliveries': copy.deepcopy(self.courier_deliveries)
        }, self.vehicle_capacity, self.opening_cost_route)
        new_state.unassigned_nodes = copy.deepcopy(self.unassigned_nodes)
        return new_state

    def get_hub_costs(self):
        return sum(hub['cost'] * hub['opened'] for hub in self.hubs)

    def get_route_costs(self):
        route_cost_total = 0
        for hub in self.hubs:
            for route in self.routes[hub['id']]:
                if len(route) > 1:  # Ensure the route has at least two nodes
                    route_cost = calculate_distance(route[0]['location'], route[1]['location'])
                    for i in range(1, len(route) - 1):
                        route_cost += calculate_distance(route[i]['location'], route[i + 1]['location'])
                    route_cost += calculate_distance(route[-1]['location'], hub['location'])
                    route_cost_total += int(route_cost * 100)
        route_cost_total += self.opening_cost_route * sum(len(self.routes[hub['id']]) for hub in self.hubs)
        return route_cost_total

    def evaluate_solution(self):
        total_cost = self.get_hub_costs() + self.get_route_costs()

        # Add courier delivery costs
        for hub_id, courier_deliveries in self.courier_deliveries.items():
            for commodity_deliveries in courier_deliveries.values():
                for courier, customer in commodity_deliveries:
                    distance_to_hub = calculate_distance(
                        next(hub['location'] for hub in self.hubs if hub['id'] == hub_id), courier['location'])
                    total_cost += distance_to_hub * 100
                    total_cost += courier['cost_per_unit'] * customer['demand']

        return total_cost

    def cost(self):
        return self.evaluate_solution()

    def objective(self):
        return self.evaluate_solution()


    def cost(self):
        return self.evaluate_solution()

    def objective(self):
        return self.evaluate_solution()

def generate_routes_with_couriers(nodes, vehicle_capacity, courier_deliveries, hub_location, distance_threshold=50):
    routes = []
    current_route = []
    current_load = {commodity: 0 for commodity in range(num_commodities)}
    visited_nodes = set()  # Global visited set to ensure each customer is visited only once

    def add_current_route():
        # Ensure that only routes with customers are added
        if any(node['type'] == 'customer' for node in current_route):
            routes.append([{'location': hub_location, 'type': 'hub'}] + current_route + [{'location': hub_location, 'type': 'hub'}])
            print(f"[VALID ROUTE] Added route with customers: {[(node['id'], node['type']) for node in current_route]}")
        else:
            print(f"[INVALID ROUTE] Skipped route without customers: {[(node['id'], node['type']) for node in current_route]}")
        current_route.clear()
        for commodity in current_load:
            current_load[commodity] = 0

    # Handle courier deliveries first
    for commodity_deliveries in courier_deliveries.values():
        for courier, customer in commodity_deliveries:
            if customer['id'] in visited_nodes:
                continue
            distance = calculate_distance(courier['location'], customer['location'])
            if distance < distance_threshold and sum(current_load.values()) + customer['demand'] <= vehicle_capacity:
                print(f"Adding customer {customer['id']} to route from courier {courier['id']}")
                current_route.append(customer)
                current_load[customer['commodity']] += customer['demand']
                visited_nodes.add(customer['id'])
                add_current_route()

    # Handle regular customer deliveries
    for node in nodes:
        if node['id'] in visited_nodes or node['type'] != 'customer':
            continue

        if sum(current_load.values()) + node['demand'] <= vehicle_capacity:
            print(f"Adding customer {node['id']} to current route")
            current_route.append(node)
            current_load[node['commodity']] += node['demand']
            visited_nodes.add(node['id'])
        else:
            add_current_route()
            current_route.append(node)
            current_load[node['commodity']] = node['demand']
            visited_nodes.add(node['id'])

    # Final check to add the last route if it's valid
    add_current_route()

    # Validate all routes one more time
    print("\n--- Final Route Validation ---")
    valid_routes = []
    for idx, route in enumerate(routes):
        customers = [node for node in route if node['type'] == 'customer']
        if customers:
            print(f"Route {idx} is valid with customers: {[node['id'] for node in customers]}")
            valid_routes.append(route)
        else:
            print(f"Route {idx} is invalid (no customers) and will be removed")

    print("\nSummary of Valid Routes:")
    for idx, route in enumerate(valid_routes):
        customers = [node['id'] for node in route if node['type'] == 'customer']
        print(f"Valid Route {idx}: Customers {customers}")

    return valid_routes







    

# Define the degree of destruction
degree_of_destruction = 0.5 

def random_removal_operator(current, rnd_state):
    solution = current.copy()
    all_customers = [node for hub_routes in solution.routes.values() for route in hub_routes for node in route if node['type'] == 'customer']
    customers_to_remove = min(int(len(all_customers) * degree_of_destruction), len(all_customers))
    
    if customers_to_remove == 0:
        return solution
    
    nodes_to_remove = rnd_state.choice(all_customers, customers_to_remove, replace=False)
    
    for node in nodes_to_remove:
        for hub_id, hub_routes in solution.routes.items():
            for route in hub_routes:
                if node in route:
                    route.remove(node)
                    if node in solution.allocations[hub_id]:
                        solution.allocations[hub_id].remove(node)
                        solution.hub_loads[hub_id][node['commodity']] -= node['demand']
                    else:
                        for courier, customer in solution.courier_deliveries[hub_id][node['commodity']]:
                            if customer == node:
                                solution.courier_deliveries[hub_id][node['commodity']].remove((courier, customer))
                                break
                    solution.unassigned_nodes.append(node)
                    break
    print(f"Random Removal Operator")    
    return solution

def worst_removal_operator(current, rnd_state):
    solution = current.copy()
    customers_to_remove = int(len([node for hub_routes in current.routes.values() for route in hub_routes for node in route]) * degree_of_destruction)
    all_customers = [(node, calculate_distance(node['location'], hub['location'])) 
                     for hub in solution.hubs for hub_routes in solution.routes[hub['id']] for route in hub_routes for node in route if node['type'] == 'customer']
    all_customers.sort(key=lambda x: x[1], reverse=True)
    nodes_to_remove = [node for node, _ in all_customers[:customers_to_remove]]

    for node in nodes_to_remove:
        for hub_id, hub_routes in solution.routes.items():
            for route in hub_routes:
                if node in route:
                    route.remove(node)
                    if node in solution.allocations[hub_id]:
                        solution.allocations[hub_id].remove(node)
                        solution.hub_loads[hub_id][node['commodity']] -= node['demand']
                    else:
                        for courier, customer in solution.courier_deliveries[hub_id][node['commodity']]:
                            if customer == node:
                                solution.courier_deliveries[hub_id][node['commodity']].remove((courier, customer))
                                break
                    solution.unassigned_nodes.append(node)
                    break
    print(f"Worst Removal Operator")
    return solution

def string_removal_operator(current, rnd_state):
    solution = current.copy()
    customers_to_remove = int(len([node for hub_routes in current.routes.values() for route in hub_routes for node in route if node['type'] == 'customer']) * degree_of_destruction)
    
    if customers_to_remove == 0:
        return solution

    hub_id = rnd_state.choice(list(solution.routes.keys()))
    route_idx = rnd_state.choice(range(len(solution.routes[hub_id])))

    route = solution.routes[hub_id][route_idx]
    customer_route = [node for node in route if node['type'] == 'customer']
    start_idx = rnd_state.randint(0, len(customer_route) - customers_to_remove)
    string_to_remove = customer_route[start_idx:start_idx + customers_to_remove]

    for node in string_to_remove:
        route.remove(node)
        if node in solution.allocations[hub_id]:
            solution.allocations[hub_id].remove(node)
            solution.hub_loads[hub_id][node['commodity']] -= node['demand']
        else:
            for courier, customer in solution.courier_deliveries[hub_id][node['commodity']]:
                if customer == node:
                    solution.courier_deliveries[hub_id][node['commodity']].remove((courier, customer))
                    break
        solution.unassigned_nodes.append(node)
    print(f"String Removal Operator")
    return solution

def greedy_insert_operator(current, rnd_state, distance_threshold=10):
    solution = current.copy()
    unassigned_nodes = [node for node in solution.unassigned_nodes if 'demand' in node]
    unassigned_nodes.sort(key=lambda x: x['demand'], reverse=True)

    assigned_customers = set()

    for node in unassigned_nodes:
        if node['id'] in assigned_customers:
            continue

        if 'commodity' not in node or node['commodity'] is None:
            continue  

        closest_hub = min(
            (hub for hub in solution.hubs if sum(solution.hub_loads[hub['id']][node['commodity']] + node['demand'] <= hub['capacity']) and sum(node['demand'] for node in solution.allocations[hub['id']]) + node['demand'] <= vehicle_capacity),
            key=lambda hub: calculate_distance(node['location'], hub['location']),
            default=None
        )

        if closest_hub:
            hub_id = closest_hub['id']
            solution.allocations[hub_id].append(node)
            solution.hub_loads[hub_id][node['commodity']] += node['demand']
            solution.unassigned_nodes.remove(node)
            assigned_customers.add(node['id'])

            closest_courier = min(
                (courier for courier in solution.courier_deliveries[hub_id][node['commodity']] if calculate_distance(courier['location'], node['location']) < distance_threshold),
                key=lambda courier: calculate_distance(courier['location'], node['location']),
                default=None
            )

            if closest_courier:
                courier_distance = calculate_distance(closest_courier['location'], node['location'])
                if courier_distance < distance_threshold:
                    hub_remaining_quantity = solution.hub_loads[hub_id][closest_courier['commodity']]
                    if hub_remaining_quantity >= node['demand']:
                        if sum(current_load.values()) + node['demand'] <= vehicle_capacity:
                            solution.courier_deliveries[hub_id][node['commodity']].append((closest_courier, node))
                            solution.unassigned_nodes.remove(node)
                            assigned_customers.add(node['id'])
                        else:
                            print(f"Skipping courier assignment for node {node['id']} due to vehicle capacity exceeded.")
                    else:
                        print(f"Skipping courier assignment for node {node['id']} due to insufficient hub quantity.")
                else:
                    print(f"Skipping courier assignment for node {node['id']} due to distance {courier_distance:.2f} exceeding threshold {distance_threshold}")
    print(f"Greedy Insertion Operator: greedy insert operator")
    return solution

def regret_insert_operator(current, rnd_state, vehicle_capacity, distance_threshold=10):
    solution = current.copy()
    unassigned_nodes = [node for node in solution.unassigned_nodes if 'demand' in node]
    if not unassigned_nodes:
        return solution

    assigned_customers = set()

    def calculate_regret(node, hubs):
        distances = [calculate_distance(node['location'], hub['location']) for hub in hubs]
        distances.sort()
        return distances[1] - distances[0]

    unassigned_nodes.sort(key=lambda node: calculate_regret(node, solution.hubs), reverse=True)

    for node in unassigned_nodes:
        if node['id'] in assigned_customers:
            continue

        if 'commodity' not in node or node['commodity'] is None:
            continue  

        closest_courier = None
        min_distance = float('inf')

        for hub_cd in solution.courier_deliveries.values():
            for courier_deliveries in hub_cd.values():
                for courier, customer in courier_deliveries:
                    distance = calculate_distance(node['location'], courier['location'])
                    if distance < min_distance:
                        min_distance = distance
                        closest_courier = courier

        if closest_courier and min_distance < distance_threshold:
            closest_hub = min((hub for hub in solution.hubs),
                              key=lambda hub: calculate_distance(closest_courier['location'], hub['location']),
                              default=None)
            if closest_hub:
                solution.courier_deliveries[closest_hub['id']][node['commodity']].append((closest_courier, node))
                solution.unassigned_nodes.remove(node)
                assigned_customers.add(node['id'])
                continue

        new_hub = min((h for h in solution.hubs if sum(solution.hub_loads[h['id']].values()) + node['demand'] <= h['capacity']),
                      key=lambda h: calculate_distance(node['location'], h['location']),
                      default=None)
        if new_hub:
            new_hub_id = new_hub['id']
            for route in solution.routes[new_hub_id]:
                if sum(n.get('demand', 0) for n in route) + node['demand'] <= vehicle_capacity and node['id'] not in [n.get('id') for n in route]:
                    route.append(node)
                    solution.allocations[new_hub_id].append(node)
                    solution.hub_loads[new_hub_id][node['commodity']] += node['demand']
                    solution.unassigned_nodes.remove(node)
                    assigned_customers.add(node['id'])
                    break
            else:
                solution.routes[new_hub_id].append([node])
                solution.allocations[new_hub_id].append(node)
                solution.hub_loads[new_hub_id][node['commodity']] += node['demand']
                solution.unassigned_nodes.remove(node)
                assigned_customers.add(node['id'])
    print("Regret Insertion Operator: regret insert operator")
    return solution


def plot_solution(solution, nodes, save_path=None):
    plt.figure(figsize=(12, 8))

    # Plot the regular routes for each hub
    for hub in solution.hubs:
        hub_id = hub['id']
        hub_location = hub['location']
        for route_idx, route in enumerate(solution.routes[hub_id]):
            if route:
                route_with_depot = [hub_location] + [node['location'] for node in route] + [hub_location]
                plt.plot(*zip(*route_with_depot), marker='o', label=f"Route {route_idx} for Depot {hub['id']}")

                quantities = {commodity: 0 for commodity in range(num_commodities)}
                pickup_quantities = {commodity: 0 for commodity in range(num_commodities)}
                delivery_quantities = {commodity: 0 for commodity in range(num_commodities)}

                for node in route:
                    if node['type'] in ['customer', 'courier'] and 'commodity' in node and 'demand' in node:
                        quantities[node['commodity']] += node['demand']
                        pickup_quantities[node['commodity']] += node['demand']

                print(f"Starting quantities for Route {route_idx} from Depot {hub_id}: {pickup_quantities}")

                for i, node in enumerate(route):
                    current_pickup_quantities = {commodity: 0 for commodity in range(num_commodities)}
                    current_delivery_quantities = {commodity: 0 for commodity in range(num_commodities)}

                    if node['type'] == 'customer' and 'commodity' in node and 'demand' in node:
                        current_delivery_quantities[node['commodity']] = node['demand']
                        quantities[node['commodity']] -= node['demand']
                    elif node['type'] == 'courier' and 'commodity' in node and 'demand' in node:
                        current_pickup_quantities[node['commodity']] = node['demand']
                        quantities[node['commodity']] += node['demand']

                    plt.scatter(*node['location'], c='b', marker='o', s=100)
                    if 'id' in node:
                        plt.text(node['location'][0], node['location'][1], str(node['id']), fontsize=12)

                    if i < len(route) - 1:
                        start = node['location']
                        end = route[i + 1]['location']
                        if start != end:  
                            print(f"  {start} -> {end}: Quantities {quantities}, Pickup {current_pickup_quantities}, Delivery {current_delivery_quantities}")

                if route:
                    last_node = route[-1]
                    end = hub_location
                    start = last_node['location']
                    if start != end:  
                        print(f"  {start} -> {end}: Quantities {quantities}, Pickup {current_pickup_quantities}, Delivery {current_delivery_quantities}")

                returned_quantities = {commodity: quantities[commodity] for commodity in range(num_commodities)}
                print(f"Ending quantities for Route {route_idx} to Depot {hub_id}: {returned_quantities}")

    # Plot the courier deliveries
    for hub_id, deliveries in solution.courier_deliveries.items():
        for commodity, delivery in deliveries.items():
            for courier_node, customer in delivery:
                # Plot the courier's route to the customer
                plt.scatter(*courier_node['location'], c='g', marker='x', s=100, label=f"Courier {courier_node.get('id', '')}")
                plt.scatter(*customer['location'], c='c', marker='*', s=100)
                plt.plot([courier_node['location'][0], customer['location'][0]], [courier_node['location'][1], customer['location'][1]], 'g--')

                # Plot the route from the hub to the courier
                hub_location = next(hub['location'] for hub in solution.hubs if hub['id'] == hub_id)
                plt.plot([hub_location[0], courier_node['location'][0]], [hub_location[1], courier_node['location'][1]], 'r--')

                if 'id' in customer:
                    plt.text(customer['location'][0], customer['location'][1], str(customer['id']), fontsize=12)

                print(f"Courier delivery from Courier {courier_node.get('id', '')} to Customer {customer.get('id', '')}:")
                print(f"  Commodity {commodity}, Quantity {customer['demand']}")

    # Plot the depot locations
    for hub in solution.hubs:
        hub_location = hub['location']
        plt.scatter(*hub_location, c='r', marker='s', s=200, label=f"Depot {hub['id']}")

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Location Routing Problem Solution with Courier Deliveries")
    plt.legend()

    if save_path:
        plt.savefig(save_path)  # Save the plot to the specified file path
    plt.close()


class LocationRoutingProblem:
    def __init__(self, num_customers, num_depots, depot_coordinates, customer_coordinates, vehicle_capacity, depot_capacities, customers_demands, opening_costs_depots, opening_cost_route):
        self.num_customers = num_customers
        self.num_depots = num_depots
        self.depot_coordinates = depot_coordinates
        self.customer_coordinates = customer_coordinates
        self.vehicle_capacity = vehicle_capacity
        self.depot_capacities = depot_capacities
        self.customers_demands = customers_demands
        self.opening_costs_depots = opening_costs_depots
        self.opening_cost_route = opening_cost_route
        self.nodes = create_nodes(customers_demands, customer_coordinates)
        self.potential_hub_locations = create_potential_hub_locations(depot_coordinates, opening_costs_depots, depot_capacities)
        self.courier_nodes = create_courier_nodes(0, max_demand=20, max_cost=5.0, grid_size=10)
    
    def initialize_solution(self):
        # Pass all the required arguments to the global initialize_solution function
        return initialize_solution(self.nodes + self.courier_nodes, self.potential_hub_locations, self.depot_capacities, self.vehicle_capacity, self.opening_cost_route)
import re
import re

def parse_courier_data(file_path):
    couriers = []
    
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the data by each courier entry
    entries = content.strip().split('--------------------------------------------------')
    
    for entry in entries:
        if entry.strip():
            # Extract courier location
            location_match = re.search(r'Courier Location: \((\d+), (\d+)\)', entry)
            if location_match:
                courier_location = (int(location_match.group(1)), int(location_match.group(2)))
            else:
                continue  # Skip this entry if location data is missing

            # Extract area covered
            area_covered_match = re.search(r'Area Covered: (\d+)', entry)
            if area_covered_match:
                area_covered = int(area_covered_match.group(1))
            else:
                continue  # Skip this entry if area covered data is missing

            # Extract closest customers' coordinates
            customers_match = re.search(r'Closest Customers\' Coordinates: \((.+?)\)', entry)
            if customers_match:
                customers_coords = [
                    tuple(map(int, coord.split(','))) for coord in re.findall(r'\((\d+), (\d+)\)', customers_match.group(1))
                ]
            else:
                customers_coords = []  # No customers found

            couriers.append({
                'location': courier_location,
                'area_covered': area_covered,
                'closest_customers': customers_coords
            })
    
    return couriers

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_solution_with_courier_and_customers(solution, nodes, couriers_data, save_path=None):
    plt.figure(figsize=(12, 8))

    # Plot the regular routes for each hub
    for hub in solution.hubs:
        hub_id = hub['id']
        hub_location = hub['location']
        for route_idx, route in enumerate(solution.routes[hub_id]):
            if route:
                route_with_depot = [hub_location] + [node['location'] for node in route] + [hub_location]
                plt.plot(*zip(*route_with_depot), marker='o', label=f"Route {route_idx} for Depot {hub['id']}")

    # Plot the depot locations
    for hub in solution.hubs:
        hub_location = hub['location']
        plt.scatter(*hub_location, c='r', marker='s', s=200, label=f"Depot {hub['id']}")

    # Plot the couriers and their closest customers
    for courier_idx, courier in enumerate(couriers_data):
        location = courier['location']
        area_covered = courier['area_covered']
        closest_customers = courier['closest_customers']
        
        # Plot courier location
        plt.scatter(location[0], location[1], c='blue', marker='D', s=100, label=f'Courier {courier_idx + 1}')
        plt.text(location[0], location[1], f'Courier {courier_idx + 1}', fontsize=12, verticalalignment='bottom', horizontalalignment='right', color='blue')

        # Draw a circle representing the courier's area of work
        circle = patches.Circle(location, area_covered, edgecolor='blue', facecolor='none', linestyle='--', linewidth=2)
        plt.gca().add_patch(circle)

        # Plot the closest customers and draw lines to the courier
        for customer_location in closest_customers:
            plt.scatter(customer_location[0], customer_location[1], c='green', marker='*', s=150, label=f'Closest Customer for Courier {courier_idx + 1}')
            plt.plot([location[0], customer_location[0]], [location[1], customer_location[1]], 'green', linestyle='--')

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Location Routing Problem Solution with Courier and Customer Locations")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.close()


def extract_total_price(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Regular expression to find all occurrences of "Total Price"
    total_price_regex = re.compile(r"Total Price \(1 Kg \* New Customer Quantity\): (\d+)")
    total_prices = total_price_regex.findall(content)

    # Convert the extracted price values to integers and calculate the sum
    total_sum = sum(int(price) for price in total_prices)

    return total_sum

@app.route('/compute-route', methods=['POST'])
def compute_route():
    num_customers, num_depots, depot_coordinates, customer_coordinates, vehicle_capacity, depot_capacities, customers_demands, opening_costs_depots, opening_cost_route = read_data_1(file_path)
    print(f"Read data: num_customers={num_customers}, num_depots={num_depots}")
    
    # Parameters
    num_couriers = 0
    max_demand = 50
    max_cost = 1
    grid_size = 10
    print(f"Parameters: num_couriers={num_couriers}, max_demand={max_demand}, max_cost={max_cost}, grid_size={grid_size}")

    # Create nodes and potential hub locations
    nodes = create_nodes(customers_demands, customer_coordinates)
    print(f"Created nodes: {nodes}")

    potential_hub_locations = create_potential_hub_locations(depot_coordinates, opening_costs_depots, depot_capacities)
    print(f"Created potential hub locations: {potential_hub_locations}")

    # Initialize the Location Routing Problem
    problem = LocationRoutingProblem(num_customers, num_depots, depot_coordinates, customer_coordinates, vehicle_capacity, depot_capacities, customers_demands, opening_costs_depots, opening_cost_route)
    print("Initialized LocationRoutingProblem")

    # Create courier nodes
    courier_nodes = create_courier_nodes(num_couriers, max_demand, max_cost, grid_size)
    print(f"Created courier nodes: {courier_nodes}")

    # Generate the initial solution
    initial_solution = problem.initialize_solution()
    print(f"Initial solution: {initial_solution}")

    print_initial_quantities(nodes + potential_hub_locations)

    # Initialize the LrpState with all required arguments
    initial_state = LrpState(initial_solution, vehicle_capacity, opening_cost_route)
    print("Initialized LrpState")

    # Define ALNS setup
    SEED = 1234
    alns = ALNS(np.random.RandomState(SEED))
    alns.add_destroy_operator(random_removal_operator)
    alns.add_destroy_operator(worst_removal_operator)
    alns.add_destroy_operator(string_removal_operator)
    alns.add_repair_operator(lambda sol, rnd: greedy_insert_operator(sol, rnd, vehicle_capacity))
    alns.add_repair_operator(lambda sol, rnd: regret_insert_operator(sol, rnd, vehicle_capacity))

    # Define selection, acceptance, and stopping criteria
    select = RouletteWheel([25, 5, 1, 0], 0.8, 1, 1)
    accept = RecordToRecordTravel.autofit(initial_state.evaluate_solution(), 0.01, 0, 5000)
    stop = MaxIterations(5000)  

    print("Starting ALNS iteration")
    result = alns.iterate(initial_state, select, accept, stop)
    print("ALNS iteration completed")

    # Get the best solution and its objective value
    solution = result.best_state
    objective = solution.evaluate_solution()
    print(f"Best heuristic objective: {objective}")

    # Plot the results
    plot_solution(solution, nodes, 'LRP.png') 

    # Ensure these methods exist in your LrpState class
    hub_costs = solution.get_hub_costs()  # Now correctly implemented
    route_costs = solution.get_route_costs()  # Now correctly implemented
    objective = solution.evaluate_solution()
     # Extract total price from courier data file
    total_price = extract_total_price("courier_data.txt")

    # Prepare the response
    response = {
        'hub_costs': hub_costs,
        'route_costs': route_costs,
        'objective': objective,
        'total_price': total_price 
    }

    return jsonify(response)

from flask import Flask, jsonify

@app.route('/compute-route_1', methods=['POST'])
def compute_route_1():
    # Read necessary data
    num_customers, num_depots, depot_coordinates, customer_coordinates, vehicle_capacity, depot_capacities, customers_demands, opening_costs_depots, opening_cost_route = read_data_1(file_path)
    print(f"Read data: num_customers={num_customers}, num_depots={num_depots}")
    
    # Parameters
    num_couriers = 0
    max_demand = 50
    max_cost = 1
    grid_size = 10
    print(f"Parameters: num_couriers={num_couriers}, max_demand={max_demand}, max_cost={max_cost}, grid_size={grid_size}")

    # Create nodes and potential hub locations
    nodes = create_nodes(customers_demands, customer_coordinates)
    print(f"Created nodes: {nodes}")

    potential_hub_locations = create_potential_hub_locations(depot_coordinates, opening_costs_depots, depot_capacities)
    print(f"Created potential hub locations: {potential_hub_locations}")

    # Initialize the Location Routing Problem
    problem = LocationRoutingProblem(num_customers, num_depots, depot_coordinates, customer_coordinates, vehicle_capacity, depot_capacities, customers_demands, opening_costs_depots, opening_cost_route)
    print("Initialized LocationRoutingProblem")

    # Create courier nodes
    courier_nodes = create_courier_nodes(num_couriers, max_demand, max_cost, grid_size)
    print(f"Created courier nodes: {courier_nodes}")

    # Generate the initial solution
    initial_solution = problem.initialize_solution()
    print(f"Initial solution: {initial_solution}")

    print_initial_quantities(nodes + potential_hub_locations)

    # Initialize the LrpState with all required arguments
    initial_state = LrpState(initial_solution, vehicle_capacity, opening_cost_route)
    print("Initialized LrpState")

    # Define ALNS setup
    SEED = 1234
    alns = ALNS(np.random.RandomState(SEED))
    alns.add_destroy_operator(random_removal_operator)
    alns.add_destroy_operator(worst_removal_operator)
    alns.add_destroy_operator(string_removal_operator)
    alns.add_repair_operator(lambda sol, rnd: greedy_insert_operator(sol, rnd, vehicle_capacity))
    alns.add_repair_operator(lambda sol, rnd: regret_insert_operator(sol, rnd, vehicle_capacity))

    # Define selection, acceptance, and stopping criteria
    select = RouletteWheel([25, 5, 1, 0], 0.8, 1, 1)
    accept = RecordToRecordTravel.autofit(initial_state.evaluate_solution(), 0.01, 0, 5000)
    stop = MaxIterations(5000)  

    print("Starting ALNS iteration")
    result = alns.iterate(initial_state, select, accept, stop)
    print("ALNS iteration completed")

    # Get the best solution and its objective value
    solution = result.best_state
    objective = solution.evaluate_solution()
    print(f"Best heuristic objective: {objective}")

    # Plot the main solution
    plot_solution(solution, nodes, 'LRP.png') 

    # Extract and plot couriers with customers
    couriers_data = parse_courier_data("courier_data.txt")
    plot_solution_with_courier_and_customers(solution, nodes, couriers_data, save_path='LRP_with_courier_and_customers.png')

    # Send a success message
    return "Operation completed successfully!", 200



    # Prepare the response
   


@app.route('/get-plot', methods=['GET'])
def get_plot():
    return send_file('LRP_with_courier_and_customers.png', mimetype='image/png')



if __name__ == '__main__':
    app.run(debug=True)