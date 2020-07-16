import csv
import random
import sys
import math
import copy
import json
import datetime
import timeit


ITEMS = []
CAPACITY = 822
MAX_ITERATIONS = 100
POP_SIZE = 100
OPTIMAL = 997

class GA:
    def __init__(self, configuration, selection_method, mutation_ratio, crossover_ratio, crossover_method, mutation_method):
        self.configuration = configuration
        self.selection_method = selection_method
        self.mutation_ratio = float(mutation_ratio)
        self.crossover_ratio = float(crossover_ratio)
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.population = self.spawn()
        self.fitvalues = []


    def spawn(self):
        population = []

        for x in range(POP_SIZE):
            population.append([random.randint(0,1) for y in range (0,len(ITEMS))])

        return population


    def fitness(self):
        fitness = []
        for i in range(len(self.population)):
            sum_w = CAPACITY + 1          
            while(sum_w > CAPACITY):
                sum_w = 0
                sum_v = 0
                ones = [] 
                for j in range(len(self.population[i])):                   
                    if(self.population[i][j] == 1):
                        sum_w += int(ITEMS[j][1])
                        sum_v += int(ITEMS[j][2])
                        ones.append(j)          
                if(sum_w > CAPACITY):
                    self.population[i][ones[random.randint(0, len(ones)-1)]] = 0
            fitness.append(sum_v)

        return fitness

    
    def rouletteWheel(self):
        sum_fit = sum(self.fitvalues)

        mate1 = []
        fit1 = 0
        mate2 = []
        num = random.randint(0, sum_fit)
        count = 0

        for i in range(len(self.population)):
            count += self.fitvalues[i]
            if(count >= num):
                mate1 = self.population.pop(i)
                fit1 = self.fitvalues.pop(i)
                break

        sum_fit = sum(self.fitvalues)
        num = random.randint(0, sum_fit)
        count = 0
            
        for i in range(len(self.population)):
            count += self.fitvalues[i]
            if(count >= num):
                mate2 = self.population[i]
                self.population.append(mate1) 
                self.fitvalues.append(fit1)           
                break

        return (mate1, mate2)

    
    def tournamentSelection(self):
        mate1 = []
        mate2 = []

        num = random.randint(0, len(self.population)-1)
        competitor1 = self.population[num]
        num2 = random.randint(0, len(self.population)-1)
        competitor2 = self.population[num2]

        if(self.fitvalues[num] > self.fitvalues[num2]):
            mate1 = competitor1
        else:
            mate1 = competitor2

        num3 = random.randint(0, len(self.population)-1)
        competitor3 = self.population[num3]
        num4 = random.randint(0, len(self.population)-1)
        competitor4 = self.population[num4]

        if(self.fitvalues[num3] > self.fitvalues[num4]):
            mate2 = competitor3
        else:
            mate2 = competitor4

        return (mate1, mate2)
    
    
    def onePointCrossover(self, mate1, mate2):
        rand = random.randint(0, len(mate1)-1)
        child1 = mate1[:rand] + mate2[rand:]
        child2 = mate2[:rand] + mate1[rand:]
        
        return (child1, child2)


    def twoPointCrossover(self, mate1, mate2):
        rand1 = random.randint(0, len(mate1)-1)
        rand2 = random.randint(0, len(mate2)-1)
        child1 = []
        child2 = []

        if(rand1 > rand2):
            temp = rand1
            rand1 = rand2
            rand2 = temp

        if(rand1 == rand2):
            (child1, child2) = self.onePointCrossover(mate1, mate2)
        else:
            child1 = mate1[:rand1] + mate2[rand1:rand2] + mate1[rand2:]
            child2 = mate2[:rand1] + mate1[rand1:rand2] + mate2[rand2:]    

        return(child1, child2)


    def bfmutation(self, individual):
        for i in range(len(individual)):
            num = random.uniform(0, 1)
            if(num <= self.mutation_ratio):
                if(individual[i] == 1):
                    individual[i] = 0
                else:
                    individual[i] = 1

        return individual

    
    def exchangeMutation(self, individual):
        num = random.uniform(0, 1)
        rand1 = 0
        rand2 = 0
        if(num <= self.mutation_ratio):
            while(rand1 == rand2):
                rand1 = random.randint(0, len(individual)-1)
                rand2 = random.randint(0, len(individual)-1)

            temp = individual[rand1]
            individual[rand1] = individual[rand2]
            individual[rand2] = temp

        return individual

 
    def inversionMutation(self, individual):
        num = random.uniform(0, 1)
        rand1 = 0
        rand2 = 0
        if(num <= self.mutation_ratio):
            while(rand1 == rand2):
                rand1 = random.randint(0, len(individual)-1)
                rand2 = random.randint(0, len(individual)-1)

            if(rand1 > rand2):
                temp = rand1
                rand1 = rand2
                rand2 = temp

            substring = list(reversed(individual[rand1:rand2-1]))
            individual = individual[:rand1] + substring + individual[rand2-1:]  

        return individual


    def insertionMutation(self, individual):
        num = random.uniform(0, 1)
        rand1 = 0
        rand2 = 0
        if(num <= self.mutation_ratio):
            while(rand1 == rand2):
                rand1 = random.randint(0, len(individual)-1)
                rand2 = random.randint(0, len(individual)-1)

            count = rand2
            target = rand1

            if(rand2 > rand1):
                while(count != target + 1):
                    temp = individual[count - 1]
                    individual[count - 1] = individual[count]
                    individual[count] = temp
                    count -= 1
            else:
                while(count != target + 1):
                    if(count == target):
                        count += 1
                        continue
                    temp = individual[count + 1]
                    individual[count + 1] = individual[count]
                    individual[count] = temp
                    count += 1

        return individual


    def insertion(self, a, count, target):
        if(count > target):
            while(count != target + 1):
                temp = a[count - 1]
                a[count - 1] = a[count]
                a[count] = temp
                count -= 1

        else:
            while(count != target + 1):
                if(count == target):
                    count += 1
                    continue
                temp = a[count + 1]
                a[count + 1] = a[count]
                a[count] = temp
                count += 1

        return a


    def displacementMutation(self, individual):
        num = random.uniform(0, 1)
        rand1 = 0
        rand2 = 0
        if(num <= self.mutation_ratio):
            while(rand1 == rand2):
                rand1 = random.randint(0, len(individual)-1)
                rand2 = random.randint(0, len(individual)-1)

            if(rand1 > rand2):
                temp = rand1
                rand1 = rand2
                rand2 = temp

            start = rand1
            end = rand2
            move = 0

            for j in range(start, end+1):
                individual = self.insertion(individual, j, move)
                move += 1
        
        return individual

    def newPop(self):
        newPop = []

        while(len(newPop) < len(self.population)):
            if(self.selection_method == "RWS"):
                (mate1, mate2) = self.rouletteWheel()
            else:
                (mate1, mate2) = self.tournamentSelection()
            num = random.uniform(0, 1)
            if(num <= self.crossover_ratio):
                if(self.crossover_method == "1PX"):
                    (child1, child2) = self.onePointCrossover(mate1, mate2)
                else:
                    (child1, child2) = self.twoPointCrossover(mate1, mate2)
                newPop.append(child1)
                newPop.append(child2)
           
        for i in range(len(newPop)):
            if(self.mutation_method == "BFM"):
                newPop[i] = self.bfmutation(newPop[i])
            elif(self.mutation_method == "EXM"):
                newPop[i] = self.exchangeMutation(newPop[i])
            elif(self.mutation_method == "IVM"):
                newPop[i] = self.inversionMutation(newPop[i])
            elif(self.mutation_method == "ISM"): 
                newPop[i] = self.insertionMutation(newPop[i])
            else:
                newPop[i] = self.displacementMutation(newPop[i])
  
        return newPop
        

    def selectBest(self):
        best = 0
        num = 0
        for i in range(len(self.fitvalues)):
            if(self.fitvalues[i] > best):
                best = self.fitvalues[i]
                num = i
        sum_w = 0
        sum_v = 0
        for j in range(len(self.population[num])):                   
                if(self.population[num][j] == 1):
                    sum_w += int(ITEMS[j][1])
                    sum_v += int(ITEMS[j][2])

        mode = 0
        for i in self.fitvalues:
            if(i == best):
                mode += 1

        knapsack = self.population[num]
                    
        return best, sum_w, mode, knapsack


    def run(self):  
        i = 1    
        generation = 0
        best = 0
        bw = 0
        m = 0
        knapsack = []
        besta = []
        bestw = []
        bestsq = []
        count = []
        self.fitvalues = self.fitness()
        currentDT = datetime.datetime.now()
        file = open("report_[GA]_" + str(currentDT)[:10] + ".txt","a") 
        file.write("Evaluation | "  + str(currentDT)[:16] + '\n')
        file.write("Configuration: " + '\t' + self.configuration + ".json\n")
        file.write('\t\t' + "GA | " + "#" + str(MAX_ITERATIONS) + " | " +  self.selection_method \
            + " | " + self.crossover_method + " (" + str(self.crossover_ratio) + ") | " + self.mutation_method \
            + " (" + str(self.mutation_ratio) + ")\n")
        file.write("=======================================================================================================\n")
        file.write("#" + '\t' + "bweight" + '\t' + "bvalue" + '\t' + "squality" + '\t' + "knapsack\n")
        file.write("-------------------------------------------------------------------------------------------------------\n")
        start = timeit.default_timer()
        while(generation < MAX_ITERATIONS):
            if(OPTIMAL in self.fitvalues):
                break
            self.population = self.newPop()
            
            self.fitvalues = self.fitness()
            generation += 1
            temp, weight, mode, k = self.selectBest()
            mode = mode/len(self.fitvalues) * 100
            s = ""
            if(best < temp):
                best = temp
            
            file.write(str(generation + 1) + '\t' + str(weight) + '\t' + str(temp) +'\t' + str(mode) + "%" + '\t' + "[" + s.join(map(str,k)) + "]\n")
            if(i == MAX_ITERATIONS or i == MAX_ITERATIONS * 0.75 or i == MAX_ITERATIONS / 2 or i == MAX_ITERATIONS / 4 ):
                count.append(i)
                besta.append(temp)
                bestw.append(weight)
                bestsq.append(mode)
            i += 1
        stop = timeit.default_timer()
        file.write("--------------------------------------------------------------------------------------------------------\n")
        file.write("[Statistics]\n")
        file.write("Runtime" + '\t' + str(stop - start) + " ms" + '\n\n')
        file.write("Convergence" + '\t' + "#" + '\t' + "bweight" + '\t'+ "bvalue" + '\t' + "squality\n")
        for i in range(len(besta)):
            file.write('\t\t' + str(count[i]) + '\t' + str(bestw[i]) + '\t\t' + str(besta[i]) + '\t\t'+ str(bestsq[i]) + "%\n")
        
        file.write('\n')
        file.write("Plateau | Longest sequence" + '\n\n')
        file.write("=========================================================================================================\n")
        file.close()

        return generation, best
        
      
class SA:
    def __init__(self, configuration, initial_temperature, cooling_rate):
        self.configuration = configuration
        self.initial_temperature = int(initial_temperature)
        self.cooling_rate = float(cooling_rate)
        self.initalsol = [random.randint(0,1) for y in range (0,len(ITEMS))]
        self.getFitness(self.initalsol, 1)
 

    def getFitness(self, solution, num):
        sum_v = 0
        sum_w = CAPACITY + 1

        if(num == 1):          
            while(sum_w > CAPACITY):
                sum_w = 0
                sum_v = 0
                ones = [] 
                for i in range(len(self.initalsol)):                   
                    if(self.initalsol[i] == 1):
                        sum_w += int(ITEMS[i][1])
                        sum_v += int(ITEMS[i][2])
                        ones.append(i)          
                if(sum_w > CAPACITY):
                    self.initalsol[ones[random.randint(0, len(ones)-1)]] = 0
        else:
            sum_w = 0
            sum_v = 0

            for i in range(len(solution)):                   
                if(solution[i] == 1):
                    sum_w += int(ITEMS[i][1])
                    sum_v += int(ITEMS[i][2])

        return sum_v

    
    def getWeight(self, solution):
        sum_w = 0

        for i in range(len(solution)):                   
            if(solution[i] == 1):
                sum_w += int(ITEMS[i][1])
        
        return sum_w

    
    def getNeighbour(self, solution):
        if(self.getWeight(solution) > CAPACITY):
            found = False
            while(found == False):
                index = random.randint(0, len(solution)-1)
                if(solution[index] == 1):
                    solution[index] = 0
                    found = True
                    return solution
        else:
            found = False
            while(found == False):
                index = random.randint(0, len(solution)-1)
                if(solution[index] == 0):
                    solution[index] = 1
                    found = True
                    return solution

    
    def run(self):
        generation = 0
        current = self.initalsol[:]
        best = current[:]
        bestCost = self.getFitness(best, 0)
        maxTemp = self.initial_temperature
        i = 1
        bw = 0
        m = 0
        knapsack = []
        besta = []
        bestw = []
        bestsq = []
        count = []
        currentDT = datetime.datetime.now()
        file = open("report_[SA]_" + str(currentDT)[:10] + ".txt","a") 
        file.write("Evaluation | "  + str(currentDT)[:16] + '\n')
        file.write("Configuration: " + '\t' + self.configuration + ".json\n")
        file.write('\t\t' + "GA | " + "#" + str(MAX_ITERATIONS) + " | " +  str(self.initial_temperature) \
            + " | " + str(self.cooling_rate) + '\n')
        file.write("=======================================================================================================\n")
        file.write("#" + '\t' + "bweight" + '\t' + "bvalue" + '\t' + "squality" + '\t' + "knapsack\n")
        file.write("-------------------------------------------------------------------------------------------------------\n")
        start = timeit.default_timer()

        while(generation < MAX_ITERATIONS and maxTemp > 0):           
            currentCost = self.getFitness(current, 0)
            if(bestCost >= OPTIMAL):
                print ("Optimal found in", generation, "generations")
                break

            neighbour = self.getNeighbour(current[:])  
            delta = (self.getFitness(neighbour, 0) - self.getFitness(current, 0))

            if(delta <= 0):
                current = neighbour[:] 
                if(self.getFitness(neighbour, 0) > self.getFitness(best, 0)):              
                    best = neighbour[:]
                    bestCost = self.getFitness(best, 0)
                    
            elif(random.uniform(0, 1) < math.exp(-1 * delta / float(maxTemp))):
                current = neighbour[:]
            maxTemp *= self.cooling_rate
            generation += 1
            s = ""
            file.write(str(generation + 1) + '\t' + str(self.getWeight(current)) + '\t' + str(currentCost) +'\t' + "100%" + '\t' + "[" + s.join(map(str,current)) + "]\n")
            if(i == MAX_ITERATIONS or i == MAX_ITERATIONS * 0.75 or i == MAX_ITERATIONS / 2 or i == MAX_ITERATIONS / 4 ):
                count.append(i)
                besta.append(currentCost)
                bestw.append(self.getWeight(current))
                bestsq.append("100")
            i += 1
        stop = timeit.default_timer()
        file.write("--------------------------------------------------------------------------------------------------------\n")
        file.write("[Statistics]\n")
        file.write("Runtime" + '\t' + str(stop - start) + " ms" + '\n\n')
        file.write("Convergence" + '\t' + "#" + '\t' + "bweight" + '\t'+ "bvalue" + '\t' + "squality\n")
        for i in range(len(besta)):
            file.write('\t\t' + str(count[i]) + '\t' + str(bestw[i]) + '\t\t' + str(besta[i]) + '\t\t'+ str(bestsq[i]) + "%\n")
        
        file.write('\n')
        file.write("Plateau | Longest sequence" + '\n\n')
        file.write("=========================================================================================================\n")
        file.close()

        return bestCost
            

class PSO:
    def __init__(self, configuration, minimum_velocity, maximum_velocity, inertia, number_particles, c1, c2):
        self.configuration = configuration
        self.minimum_velocity = minimum_velocity
        self.maximum_velocity = maximum_velocity
        self.inertia = float(inertia)
        self.number_particles = int(number_particles)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.positions = self.spawn()
        self.velocities = [[0] * len(ITEMS)] * self.number_particles
        self.fitvalues = []
        self.p_best = [[0] * len(ITEMS)] * self.number_particles
        self.p_best_fit = [0] * self.number_particles
        self.g_best = []
        self.g_best_fit = 0

    
    def spawn(self):
        positions = []

        for x in range(self.number_particles):
            positions.append([random.randint(0,1) for y in range (0,len(ITEMS))])

        return positions


    def fitness(self):
        fitness = []
        for i in range(len(self.positions)):
            sum_w = CAPACITY + 1          
            while(sum_w > CAPACITY):
                sum_w = 0
                sum_v = 0
                ones = [] 
                for j in range(len(self.positions[i])):                   
                    if(self.positions[i][j] == 1):
                        sum_w += int(ITEMS[j][1])
                        sum_v += int(ITEMS[j][2])
                        ones.append(j)          
                if(sum_w > CAPACITY):
                    self.positions[i][ones[random.randint(0, len(ones)-1)]] = 0
            fitness.append(sum_v)

        return fitness

    
    def setBest(self, num):
        if(self.fitvalues[num] > self.p_best_fit[num]):
            self.p_best[num] = copy.copy(self.positions[num])
            self.p_best_fit[num] = self.fitvalues[num]
        
        if(self.fitvalues[num] > self.g_best_fit):
            self.g_best = copy.copy(self.positions[num])
            self.g_best_fit = self.fitvalues[num]


    def update(self, num):
        rand1 = random.uniform(0, 1)
        rand2 = random.uniform(0, 1)

        for i in range(len(ITEMS)):
            self.velocities[num][i] = self.inertia * self.velocities[num][i] \
                                    + self.c1 * rand1 * (self.p_best[num][i] - self.positions[num][i]) \
                                    + self.c2 * rand2 * (self.g_best[i] - self.positions[num][i])
            rand = random.uniform(0, 1)
            if(rand < 1.0/(1.0 + math.exp((-1.0) * self.velocities[num][i]))):
                self.positions[num][i] = 1
            else:
                self.positions[num][i] = 0

    
    def selectBest(self):
        best = 0
        num = 0
        for i in range(len(self.fitvalues)):
            if(self.fitvalues[i] > best):
                best = self.fitvalues[i]
                num = i
        sum_w = 0
        sum_v = 0
        for j in range(len(self.positions[num])):                   
                if(self.positions[num][j] == 1):
                    sum_w += int(ITEMS[j][1])
                    sum_v += int(ITEMS[j][2])

        mode = 0
        for i in self.fitvalues:
            if(i == best):
                mode += 1

        knapsack = self.positions[num]
                    
        return best, sum_w, mode, knapsack


    def run(self):
        i = 1    
        generation = 0
        best = 0
        bw = 0
        m = 0
        knapsack = []
        besta = []
        bestw = []
        bestsq = []
        count = []
        self.fitvalues = self.fitness()
        currentDT = datetime.datetime.now()
        file = open("report_[PSO]_" + str(currentDT)[:10] + ".txt","a") 
        file.write("Evaluation | "  + str(currentDT)[:16] + '\n')
        file.write("Configuration: " + '\t' + self.configuration + ".json\n")
        file.write('\t\t' + "GA | " + "#" + str(self.number_particles) + " | " +  str(self.minimum_velocity) \
            + " | " + str(self.maximum_velocity) + str(self.c1) + " | " + str(self.c2) \
            + " | " + str(self.inertia) + '\n')
        file.write("=======================================================================================================\n")
        file.write("#" + '\t' + "bweight" + '\t' + "bvalue" + '\t' + "squality" + '\t' + "knapsack\n")
        file.write("-------------------------------------------------------------------------------------------------------\n")
        start = timeit.default_timer()
        self.fitvalues = self.fitness()

        for i in range(self.number_particles):
            self.setBest(i)

        while(generation < MAX_ITERATIONS):
            if(self.g_best_fit >= OPTIMAL):
                print ("Optimal found in", generation, "generations")
                break
            for i in range(self.number_particles):               
                self.update(i)
                self.fitvalues = self.fitness()
                self.setBest(i)
            generation += 1
            temp, weight, mode, k = self.selectBest()
            mode = mode/len(self.fitvalues) * 100
            s = ""

            file.write(str(generation + 1) + '\t' + str(weight) + '\t' + str(temp) +'\t' + str(mode) + "%" + '\t' + "[" + s.join(map(str,k)) + "]\n")
            if(i == MAX_ITERATIONS or i == MAX_ITERATIONS * 0.75 or i == MAX_ITERATIONS / 2 or i == MAX_ITERATIONS / 4 ):
                count.append(i)
                besta.append(temp)
                bestw.append(weight)
                bestsq.append(mode)
            i += 1
        stop = timeit.default_timer()
        file.write("--------------------------------------------------------------------------------------------------------\n")
        file.write("[Statistics]\n")
        file.write("Runtime" + '\t' + str(stop - start) + " ms" + '\n\n')
        file.write("Convergence" + '\t' + "#" + '\t' + "bweight" + '\t'+ "bvalue" + '\t' + "squality\n")
        for i in range(len(besta)):
            file.write('\t\t' + str(count[i]) + '\t' + str(bestw[i]) + '\t\t' + str(besta[i]) + '\t\t'+ str(bestsq[i]) + "%\n")
        
        file.write('\n')
        file.write("Plateau | Longest sequence" + '\n\n')
        file.write("=========================================================================================================\n")
        file.close()

        return generation, self.g_best_fit


def loadData():
    with open("../data/knapsack_instance.csv") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        index = 0
        for row in spamreader:
            if (index == 0):
                index += 1
                continue
            word = ', '.join(row)
            ITEMS.append(word.split(';'))

def main():
    loadData()

    if(sys.argv[1] == "-configuration"):
        file = sys.argv[2]
        if(file[:2] == "ga"):
            f = open("../data/ga/ga" + file[2:])
            data = json.load(f)
            selection_method = data["selection_method"]
            configuration = data["configuration"]
            mutation_ratio = data["mutation_ratio"]
            crossover_ratio = data["crossover_ratio"]
            crossover_method = data["crossover_method"]
            mutation_method = data["mutation_method"]
            ga = GA(configuration, selection_method, mutation_ratio, crossover_ratio, crossover_method, mutation_method)
            ga.run()
        elif(file[:2] == "sa"):
            f = open("../data/sa/sa" + file[2:])
            data = json.load(f)
            initial_temperature = data["initial_temperature"]
            configuration = data["configuration"]
            cooling_rate = data["cooling_rate"]
            sa = SA(configuration, initial_temperature, cooling_rate)
            sa.run()
        elif(file[:3] == "pso"):
            f = open("../data/pso/pso" + file[3:])
            data = json.load(f)
            minimum_velocity = data["minimum_velocity"]
            maximum_velocity = data["maximum_velocity"]
            inertia = data["inertia"]
            configuration = data["configuration"]
            number_particles = data["number_particles"]
            c1 = data["c1"]
            c2 = data["c2"]
            pso = PSO(configuration, minimum_velocity, maximum_velocity, inertia, number_particles, c1, c2)
            pso.run()
        else:
            print("invalid config given")
    elif(sys.argv[1] == "-search_best_configuration"):
            if(sys.argv[2] == "ga"):
                bestg = MAX_ITERATIONS
                bestv = 0
                bestj = ""
                bestvj = ""
                for i in range(28):
                    f = ""
                    if(i < 9):
                        f = open("../data/ga/ga_default_" + "0" + str(i + 1) + ".json")
                    else:
                        f = open("../data/ga/ga_default_" + str(i + 1) + ".json")
                    data = json.load(f)
                    selection_method = data["selection_method"]
                    configuration = data["configuration"]
                    mutation_ratio = data["mutation_ratio"]
                    crossover_ratio = data["crossover_ratio"]
                    crossover_method = data["crossover_method"]
                    mutation_method = data["mutation_method"]
                    ga = GA(configuration, selection_method, mutation_ratio, crossover_ratio, crossover_method, mutation_method)
                    temp, temp2 = ga.run()
                    if(temp2 > bestv):
                        bestv = temp2
                        bestvj = configuration
                    if(temp < bestg):
                        bestg = temp
                        bestj = configuration
                if(bestj != MAX_ITERATIONS):
                    print(bestj)
                    f = open("../data/ga/" + bestj + ".json")
                    data = json.load(f)
                    with open("ga_best.json", 'w') as outfile:
                        json.dump(data, outfile)
                else:
                    print(bestvj)
                    f = open("../data/ga/" + bestvj + ".json")
                    data = json.load(f)
                    with open("ga_best.json", 'w') as outfile:
                        json.dump(data, outfile)
                
            elif(sys.argv[2] == "sa"):
                bestv = 0
                bestvj = ""
                for i in range(25):
                    f = ""
                    if(i < 9):
                        f = open("../data/sa/sa_default_" + "0" + str(i + 1) + ".json")
                    else:
                        f = open("../data/sa/sa_default_" + str(i + 1) + ".json")
                    data = json.load(f)
                    initial_temperature = data["initial_temperature"]
                    configuration = data["configuration"]
                    cooling_rate = data["cooling_rate"]
                    sa = SA(configuration, initial_temperature, cooling_rate)
                    temp = sa.run()
                    if(temp > bestv):
                        bestv = temp
                        bestvj = configuration
                
                print(bestvj)
                f = open("../data/sa/" + bestvj + ".json")
                data = json.load(f)
                with open("sa_best.json", 'w') as outfile:
                    json.dump(data, outfile)

            elif(sys.argv[2] == "pso"):
                bestg = MAX_ITERATIONS
                bestv = 0
                bestj = ""
                bestvj = ""
                for i in range(25):
                    f = ""
                    if(i < 9):
                        f = open("../data/pso/pso_default_" + "0" + str(i + 1) + ".json")
                    else:
                        f = open("../data/pso/pso_default_" + str(i + 1) + ".json")
                    data = json.load(f)
                    minimum_velocity = data["minimum_velocity"]
                    maximum_velocity = data["maximum_velocity"]
                    inertia = data["inertia"]
                    configuration = data["configuration"]
                    number_particles = data["number_particles"]
                    c1 = data["c1"]
                    c2 = data["c2"]
                    pso = PSO(configuration, minimum_velocity, maximum_velocity, inertia, number_particles, c1, c2)
                    temp, temp2 = pso.run()
                    if(temp2 > bestv):
                        bestv = temp2
                        bestvj = configuration
                    if(temp < bestg):
                        bestg = temp
                        bestj = configuration
                if(bestj != MAX_ITERATIONS):
                    print(bestj)
                    f = open("../data/pso/" + bestj + ".json")
                    data = json.load(f)
                    with open("pso_best.json", 'w') as outfile:
                        json.dump(data, outfile)
                else:
                    print(bestvj)
                    f = open("../data/pso/" + bestvj + ".json")
                    data = json.load(f)
                    with open("pso_best.json", 'w') as outfile:
                        json.dump(data, outfile)
            else:
                print("incorrect config given")
    
if __name__ == "__main__":
    main()