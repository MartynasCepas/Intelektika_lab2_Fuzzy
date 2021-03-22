import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Generate universe variables
# * Manufactured on year on ranges [1990, 2020]
# * Mileage on ranges [50 000, 500 000]
# * Engine power in kilowatts [80, 300]
# * RESULT: Probability to sell a car [0, 100] in units of percentage points

x_year = np.arange(1990, 2020, 1)
x_mileage = np.arange(50000, 500000, 1)
x_power = np.arange(80, 300, 1)
x_sold = np.arange(0, 101, 1)

# Generate fuzzy membership functions
# Manufactured on year
year_lo = fuzz.trapmf(x_year, [1990, 1990, 2000, 2005])
year_md = fuzz.trapmf(x_year, [2000, 2005, 2010, 2015])
year_hi = fuzz.trapmf(x_year, [2010, 2015, 2020, 2020])

# Mileage
mileage_lo = fuzz.trapmf(x_mileage, [50000, 50000, 150000, 200000])
mileage_md = fuzz.trapmf(x_mileage, [150000, 200000, 300000, 350000])
mileage_hi = fuzz.trapmf(x_mileage, [300000, 350000, 500000, 500000])

# Power
power_lo = fuzz.trapmf(x_power, [80, 80, 100, 150])
power_md = fuzz.trapmf(x_power, [100, 150, 170, 220])
power_hi = fuzz.trapmf(x_power, [170, 220, 300, 300])

# RESULT
sold_lo = fuzz.trapmf(x_sold, [0, 0, 20, 40])
sold_md = fuzz.trimf(x_sold, [20, 50, 80])
sold_hi = fuzz.trapmf(x_sold, [60, 85, 100, 100])

# Visualize these universes and membership functions
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(8, 12))
ax0.plot(x_year, year_lo, 'aqua', linewidth=2, label='Old')
ax0.plot(x_year, year_md, 'aquamarine', linewidth=2, label='Average')
ax0.plot(x_year, year_hi, 'lightgreen', linewidth=2, label='New')
ax0.set_title('Year of manufactoring')
ax0.legend()

ax1.plot(x_mileage, mileage_lo, 'aqua', linewidth=1.5, label='Low')
ax1.plot(x_mileage, mileage_md, 'aquamarine', linewidth=1.5, label='Average')
ax1.plot(x_mileage, mileage_hi, 'lightgreen', linewidth=1.5, label='High')
ax1.set_title('Mileage')
ax1.legend()

ax2.plot(x_power, power_lo, 'aqua', linewidth=2, label='Low')
ax2.plot(x_power, power_md, 'aquamarine', linewidth=2, label='Medium')
ax2.plot(x_power, power_hi, 'lightgreen', linewidth=2, label='High')
ax2.set_title('Kilowatts')
ax2.legend()

ax3.plot(x_sold, sold_lo, 'aqua', linewidth=2, label='Small')
ax3.plot(x_sold, sold_md, 'aquamarine', linewidth=2, label='Average')
ax3.plot(x_sold, sold_hi, 'lightgreen', linewidth=2, label='High')
ax3.set_title('Probability of selling a car')
ax3.legend()

#Turn off top/right axes
for ax in (ax0, ax1, ax2, ax3):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

plt.show()

#Set input data
year = 2016
mileage = 250000
power = 260

# We need the set activation of our fuzzy membership functions at these values.
# Using fuzz.interp_membership

year_level_lo = fuzz.interp_membership(x_year, year_lo, year)
year_level_md = fuzz.interp_membership(x_year, year_md, year)
year_level_hi = fuzz.interp_membership(x_year, year_hi, year)

mileage_level_lo = fuzz.interp_membership(x_mileage, mileage_lo, mileage)
mileage_level_md = fuzz.interp_membership(x_mileage, mileage_md, mileage)
mileage_level_hi = fuzz.interp_membership(x_mileage, mileage_hi, mileage)

power_level_lo = fuzz.interp_membership(x_power, power_lo, power)
power_level_md = fuzz.interp_membership(x_power, power_md, power)
power_level_hi = fuzz.interp_membership(x_power, power_hi, power)

# Additional calculations
not_small_power = np.fmax(power_level_md, power_level_hi)
not_high_power = np.fmax(power_level_lo, power_level_md)
not_old_car = np.fmax(year_level_md, year_level_hi)
not_new_car = np.fmax(year_level_lo, year_level_md)
not_high_mileage = np.fmax(mileage_level_lo, mileage_level_md)
not_low_mileage = np.fmax(mileage_level_md, mileage_level_hi)

# Rules for small selling probability
# 1. Old car and mileage high
# 2. Old car and low power
# 3. low mileage and low power

active_rule1 = np.fmin(year_level_hi, mileage_level_hi)
active_rule2 = np.fmin(year_level_hi, mileage_level_lo)
active_rule3 = np.fmin(mileage_level_lo, power_level_lo)

small_probability = np.fmax(active_rule1, np.fmax(active_rule2, active_rule3))

# Now we apply result by clipping the top off the corresponding output
# Membership function with `np.fmin`

sold_activation_lo = np.fmin(small_probability, sold_lo)
print("Low probability for selling a car")
print(small_probability)

# Rules for average selling probability
# 1. Not old car, not high power
# 2. Not new car, not low power
# 3. Not new car, not high mileage
# 4. Medium age and medium mileage
# 5. Medium power and medium age

active_rule1 = np.fmin(not_old_car, not_high_power)
active_rule2 = np.fmin(not_new_car, not_small_power)
active_rule3 = np.fmin(not_new_car, not_high_mileage)
active_rule4 = np.fmin(year_level_md, mileage_level_md)
active_rule5 = np.fmin(power_level_md, year_level_md)

average_probability = np.fmax(active_rule1, np.fmax( active_rule2, np.fmax(active_rule3, np.fmax(active_rule4, active_rule5))))

# Now we apply result by clipping the top off the corresponding output
# membership function with `np.fmin`
sold_activation_md = np.fmin(average_probability, sold_md)

print("Average probability for selling a car")
print(average_probability)


# Rules for high selling probability
# 1. New car and high power
# 2. High power and low mileage
# 3. New car and low mileage

active_rule1 = np.fmin(year_level_hi, power_level_hi)
active_rule2 = np.fmin(power_level_hi, mileage_level_lo)
active_rule3 = np.fmin(year_level_hi, mileage_level_lo)

high_probability = np.fmax(active_rule1, np.fmax(active_rule2, active_rule3))

# Now we apply result by clipping the top off the corresponding output
# membership function with `np.fmin`

sold_activation_hi = np.fmin(high_probability, sold_hi)

print("High probability for selling a car")
print(high_probability)

sold0 = np.zeros_like(x_sold)

# Visualize results of membership activity
fig, ax0 = plt.subplots(figsize=(8, 3))
ax0.fill_between(x_sold, sold0, sold_activation_lo, facecolor='aqua', alpha=0.7)
ax0.plot(x_sold, sold_lo, 'turquoise', linewidth=1, linestyle='--', )
ax0.fill_between(x_sold, sold0, sold_activation_md, facecolor='aquamarine', alpha=0.7)
ax0.plot(x_sold, sold_md, 'blue', linewidth=1, linestyle='--')
ax0.fill_between(x_sold, sold0, sold_activation_hi, facecolor='lightgreen', alpha=0.7)
ax0.plot(x_sold, sold_hi, 'green', linewidth=1, linestyle='--')
ax0.set_title('Membership activity')

for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()

# Aggregate all three output membership functions together
aggregated = np.fmax(sold_activation_lo, np.fmax(sold_activation_md, sold_activation_hi))

# Calculate defuzzified result
sold_centroid = fuzz.defuzz(x_sold, aggregated, 'centroid')
sold_bisector = fuzz.defuzz(x_sold, aggregated, 'bisector')
sold_mom = fuzz.defuzz(x_sold, aggregated, 'mom') #mean of maximum
sold_som = fuzz.defuzz(x_sold, aggregated, 'som') #min of maximum
sold_lom = fuzz.defuzz(x_sold, aggregated, 'lom') #max of maximum

sold_activation = fuzz.interp_membership(x_sold, aggregated, sold_centroid) # for plot

# Visualize defuzzified result
fig, ax0 = plt.subplots(figsize=(8, 3))
ax0.plot(x_sold, sold_lo, 'turquoise', linewidth=1, linestyle='--', )
ax0.plot(x_sold, sold_md, 'blue', linewidth=1, linestyle='--')
ax0.plot(x_sold, sold_hi, 'green', linewidth=1, linestyle='--')
ax0.fill_between(x_sold, sold0, aggregated, facecolor='aquamarine', alpha=0.7)
ax0.plot([sold_centroid, sold_centroid], [0, sold_activation], 'black', linewidth=1.5, alpha=0.9)
ax0.plot([sold_bisector, sold_bisector], [0, sold_activation], 'darkblue', linewidth=1.5, alpha=0.9)
ax0.plot([sold_mom, sold_mom], [0, 0.4], 'darkgreen', linewidth=1.5, alpha=0.9)

ax0.set_title('Aggregated membership and results')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()

print("Defuzzified results:")
print("Defuzzification: Centroid")
print(np.round(sold_centroid,2))
print("Defuzzification: Bisector")
print(np.round(sold_bisector,2))
print("Defuzzification: Mean of maximum")
print(np.round(sold_mom,2))
print("Defuzzification: Min of maximum")
print(np.round(sold_som,2))
print("Defuzzification: Max of maximum")
print(np.round(sold_lom,2))