from flask import Flask, request, jsonify, render_template
import pandas as pd
import itertools

app = Flask(__name__)

# ================== HOME ==================
@app.route("/")
def home():
    return render_template("index.html")


# ================== FUZZY ==================
@app.route("/hitung_fuzzy", methods=["POST"])
def hitung_fuzzy():
    cabai = float(request.form["cabai"])
    sambal = float(request.form["sambal"])

    # Contoh fuzzy Sugeno linear
    hasil = (cabai * 4) + (sambal * 6)

    return render_template("index.html", hasil=hasil)


# ================== KNAPSACK (GA) ==================
@app.route("/run", methods=["POST"])
def run_ga():
    data = request.get_json()

    pop_size = int(data["pop_size"])
    generations = int(data["generations"])
    crossover_rate = float(data["crossover_rate"])
    mutation_rate = float(data["mutation_rate"])

    # ----- DATA TETAP (sesuai permintaanmu) -----
    weights = [2, 3, 4, 5, 9]
    values = [3, 4, 8, 8, 10]
    capacity = 20

    import random

    def fitness(chrom):
        total_w = sum(w * c for w, c in zip(weights, chrom))
        total_v = sum(v * c for v, c in zip(values, chrom))
        return total_v if total_w <= capacity else 0

    # INISIAL POPULASI
    population = [[random.randint(0, 1) for _ in weights] for _ in range(pop_size)]

    logs = []

    for g in range(generations):
        ranked = sorted(population, key=lambda x: fitness(x), reverse=True)
        logs.append({"gen": g, "best": ranked[0], "fitness": fitness(ranked[0])})

        next_gen = ranked[:2]  # elitism

        while len(next_gen) < pop_size:
            p1, p2 = random.sample(ranked[:4], 2)
            if random.random() < crossover_rate:
                point = random.randint(1, len(weights)-1)
                child = p1[:point] + p2[point:]
            else:
                child = p1[:]

            # mutation
            for i in range(len(child)):
                if random.random() < mutation_rate:
                    child[i] = 1 - child[i]

            next_gen.append(child)

        population = next_gen

    best = max(population, key=lambda x: fitness(x))
    total_w = sum(w * c for w, c in zip(weights, best))
    total_v = sum(v * c for v, c in zip(values, best))
    items = [i+1 for i, v in enumerate(best) if v == 1]

    return jsonify({
        "best_chromosome": best,
        "best_items": items,
        "total_weight": total_w,
        "total_value": total_v,
        "fitness": fitness(best),
        "logs": logs
    })


# ================== TSP BRUTE FORCE ==================
@app.route("/tsp", methods=["POST"])
def tsp():
    file = request.files.get("file")

    if file is None or file.filename == "":
        return jsonify({"error": "Harus upload file Excel"}), 400

    # Membaca excel sesuai format kamu (header = A,B,C,... dan index = A,B,C,...)
    df = pd.read_excel(file, header=0, index_col=0)

    cities = df.columns.tolist()
    matrix = df.values.tolist()
    n = len(cities)

    best_path = None
    best_cost = float("inf")

    # Brute-force TSP
    for perm in itertools.permutations(range(n)):
        cost = 0

        for i in range(n - 1):
            cost += matrix[perm[i]][perm[i + 1]]

        cost += matrix[perm[-1]][perm[0]]  # kembali ke kota awal

        if cost < best_cost:
            best_cost = cost
            best_path = perm

    best_route = [cities[i] for i in best_path]

    return jsonify({
        "route": best_route + [best_route[0]],
        "distance": best_cost
    })


if __name__ == "__main__":
    app.run(debug=True)
