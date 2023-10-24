import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math

# streamlit settings
st.set_page_config(layout="wide")
st.title("Sustainable aircraft designer")
# Streamlit sidebar and common constants
# add a tab system
tab1, tab2 = st.tabs(["Plot", "Electric Commercial Flight"])
with tab1:
    col1, col2 = st.columns([1,2])
    with col1:
        with st.expander("General Information", expanded=True):
            col3, col4 = st.columns(2)
            distance = col3.number_input("Distance of the flight (km)", 100, 1000, 275, 25)
            co2_per_kWh = col4.number_input(
                "CO2 emissions per kWh (kg CO2eq/kWh)",
                0.1,
                1.0,
                0.219,
                0.01,
                help="CO2eq emissions per kWh of electricity. The default value is the value used by the International Energy Agency (IEA) for Portugal in 2019.",
            )
        # Plane constants and calculations
        with st.expander("Plane settings", expanded=True):
            fuel_consumption_plane = st.number_input(
                "Fuel consumption of the airplane (kg/km)",
                1.0,
                5.0,
                2.3,
                0.1,
                help="The fuel consumption of the airplane is the amount of fuel used per km. The default value is the value of the Boeing 737-800 max.",
            )
            lto_factor = st.slider(
                "LTO factor",
                1.0,
                2.0,
                1.25,
                0.01,
                help="LTO stands for Landing and Take Off, this factor accounts for the extra fuel used during these phases of the flight. The default value is 1.25, which is the value used by the International Civil Aviation Organization (ICAO) for short haul flights.",
            )
            passenger_capacity_plane = st.slider(
                "Passenger capacity of the airplane", 50, 300, 189, 1
            )
            co2_per_kg_fuel = st.number_input(
                "CO2 emissions per kg of fuel (kg CO2/kg fuel)", 1.0, 5.0, 3.0, 0.01
            )
            fill_car = fill_plane = np.linspace(0.1, 1, 100)  # 10% to 100%
            co2_per_passenger_plane = np.zeros(fill_plane.shape)
            co2_per_passenger_plane_lto = np.zeros(fill_plane.shape)
            for i, f_plane in enumerate(fill_plane):
                co2_per_passenger_plane[i] = (
                    distance * fuel_consumption_plane * co2_per_kg_fuel
                ) / (passenger_capacity_plane * f_plane)
                co2_per_passenger_plane_lto[i] = (
                    distance * fuel_consumption_plane * co2_per_kg_fuel * lto_factor
                ) / (passenger_capacity_plane * f_plane)
        # Train constants and calculations
        with st.expander("Train settings", expanded=True):
            power_max_train = st.slider(
                "Maximum power of the train (kW)", 1000, 10000, 5600, 100
            )
            efficiency_train = st.slider(
                "Avg. % of train Power used during trip", 0.5, 1.0, 0.8, 0.05
            )
            total_carriages = st.slider("Total number of carriages in the train", 1, 20, 8, 1)
            standard_carriage_seats = st.slider(
                "Number of seats in standard carriages", 50, 200, 116, 1
            )
            first_class_carriage_seats = st.slider(
                "Number of seats in first class carriages", 50, 200, 106, 1
            )
            bar_carriage_seats = st.slider("Number of seats in bar carriages", 0, 50, 0, 1)
            speed_train = st.slider("Average speed of the train (km/h)", 50, 200, 100, 5)

            total_capacity_train = (
                (total_carriages) * standard_carriage_seats
                + first_class_carriage_seats * 2
                + bar_carriage_seats
            )
            energy_used_train_kWh = (
                power_max_train * efficiency_train * distance
            ) / speed_train
            total_co2_emissions_train = energy_used_train_kWh * co2_per_kWh
            co2_per_passenger_train = np.zeros(fill_plane.shape)
            for i, f_train in enumerate(fill_plane):
                co2_per_passenger_train[i] = total_co2_emissions_train / (
                    total_capacity_train * f_train
                )

        # Car and Tesla constants and calculations
        with st.expander("Car settings", expanded=True):
            efficiency_car = np.linspace(0.05, 0.15, 8)  # L/km, varying from 5 to 15 L/100km
            car_capacity = st.slider("Passenger capacity of the car", 1, 8, 4, 1)
            co2_per_l_fuel = st.slider(
                "CO2 emissions per liter of fuel (kg CO2/liter)", 1.0, 5.0, 2.31, 0.01
            )
            energy_consumption_tesla_range = np.array([16.5]) / 100  # kWh/km

            co2_per_passenger_car = np.zeros((efficiency_car.size, fill_car.size))
            co2_per_passenger_tesla_range = np.zeros(
                (energy_consumption_tesla_range.shape[0], fill_car.shape[0])
            )
            for j, e_tesla in enumerate(energy_consumption_tesla_range):
                energy_used_tesla_kWh = distance * e_tesla
                total_co2_emissions_tesla = energy_used_tesla_kWh * co2_per_kWh
                for i, f_car in enumerate(fill_car):
                    co2_per_passenger_tesla_range[j, i] = total_co2_emissions_tesla / (
                        f_car * car_capacity
                    )

            for i, e_car in enumerate(efficiency_car):
                for j, f_car in enumerate(fill_car):
                    co2_per_passenger_car[i, j] = (distance * e_car * co2_per_l_fuel) / (
                        f_car * car_capacity
                    )


    # Create a Plotly figure
    fig = go.Figure()

    # Add the traces to the Plotly figure
    fig.add_trace(
        go.Scatter(
            x=fill_plane,
            y=co2_per_passenger_plane_lto,
            mode="lines",
            name="Airplane (with LTO)",
            line=dict(color="lightblue", width=4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fill_plane,
            y=co2_per_passenger_plane,
            mode="lines",
            name="Airplane (without LTO)",
            line=dict(color="pink", width=4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fill_plane,
            y=co2_per_passenger_train,
            mode="lines",
            name="Electric Train",
            line=dict(color="green", width=4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fill_plane,
            y=co2_per_passenger_tesla_range[0, :],
            mode="lines",
            name="Tesla Model 3 (Long Range)",
            line=dict(color="red", width=4),
        )
    )

    # Add Tesla Model 3 traces
    for i, e_car in enumerate(efficiency_car):
        co2_per_passenger_car_at_fill = np.interp(
            fill_plane, fill_car, co2_per_passenger_car[i, :]
        )
        fig.add_trace(
            go.Scatter(
                x=fill_plane,
                y=co2_per_passenger_car_at_fill,
                mode="lines",
                name=f"Car ({e_car*100:.1f} L/100km)",
                line=dict(color=px.colors.sequential.Plasma[i]),
            )
        )

    # Add vertical lines and annotations
    for occupancy in [0.25, 0.5, 0.75]:
        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=occupancy,
                x1=occupancy,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="gray", width=1, dash="dot"),
            )
        )
        fig.add_annotation(
            x=occupancy,
            y=1,
            yref="paper",
            text=f"{int(occupancy * car_capacity)} occupants",
            showarrow=False,
            font=dict(size=14),
        )

    # Update layout
    fig.update_layout(
        title="CO2 Emissions per Passenger for Different Transport Modes (Including LTO for Airplane)",
        xaxis_title="Percentage Fill (Car (gas), Tesla Model 3, Plane, and Train)",
        yaxis_title="CO2 Emissions per Passenger (kg)",
        xaxis=dict(gridcolor="gray"),
        yaxis=dict(gridcolor="gray"),
        # make mouse hover x-line
        hovermode="x unified",
        # make height of the plot 700 pixels
        height=700,
        # make legend horizontal
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="right", x=1),
    )

    with col2:
        # Display the Plotly figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Title and Tabs
    with st.expander("About"):
        st.write(
            '''
            This section of the app is dedicated to the design of an electric commercial flight.
            The design is based on the Boeing 737-800 MAX, which is a short to medium range commercial aircraft.
            The design is based on the following assumptions:
            - The aircraft is powered by a futuristic electric turbofan engine
            - The aircraft is powered by a battery (albeit a futuristic one)
            '''
        )

    # Constants
    gravitational_constant = 9.81  # m/s^2

    # System Level Requirements
    st.header("System Level Requirements")
    col1, col2, col3 = st.columns(3)
    with col1:
        req1_speed = st.number_input("Requirement 1: Target Cruising Speed (km/h)", min_value=200, max_value=1000, value=840)
    with col2:
        req2_range = st.number_input("Requirement 2: Target Maximum Range (km)", min_value=100, max_value=8000, value=5765)
    with col3:
        req3_payload = st.number_input("Requirement 3: Target Payload (kg)", min_value=100, max_value=100000, value=20000)

    # Subsystem Level Requirements - Motor Subsystem
    st.header("eTurbine Subsystem")
    col1, col2, col3 = st.columns(3)
    with col1:
        subreq1_motor_thrust = st.number_input("Sub-Requirement 1: Target eTurbofan thrust (kN)", min_value=10.0, max_value=350.0, value=147.58)
    with col2:
        subreq2_motor_mass = st.number_input("Sub-Requirement 2: Max Motor Mass (kg)", min_value=10, max_value=10000, value=2780)
    with col3:
        heat_dissipation_capacity = st.number_input('Heat Dissipation Capacity (KW/kg, Control Point)', min_value=0.1, max_value=20.0, value=10.0)

    # Control Points (Input Widgets)
    st.header("Control Points")
    col1, col2, col3 = st.columns(3)
    with col1:
        num_engines = st.number_input('Number of Engines (Control Point)', min_value=1, max_value=8, value=2)
        lift_to_drag_ratio = st.slider('Lift-to-Drag Ratio (Control Point)', min_value=5.0, max_value=25.0, value=17.0)
    with col2:
        battery_mass_kg = st.number_input('Battery Mass (kg, Control Point)', min_value=1000, max_value=50000, value=20000)
        battery_specific_energy_Wh_kg = st.number_input('Battery Specific Energy (Wh/kg, Control Point)', min_value=100, max_value=500, value=290)
        battery_multiplier = st.slider('Battery Energy density Multiplier (Control Point)', min_value=0.5, max_value=200.0, value=40.0)
    with col3:
        eta_i = st.slider('Inverter Efficiency (Control Point)', min_value=0.5, max_value=1.0, value=0.98)
        eta_m = st.slider('Motor Efficiency (Control Point)', min_value=0.5, max_value=1.0, value=0.95, key='eta_m')
        eta_p = st.slider('Propulsive Efficiency (Control Point)', min_value=0.5, max_value=1.0, value=0.7, key='eta_p')

    st.markdown("---")

    # Boeing 737-800 MAX Constants
    b737_max_range = 5765  # km
    b737_max_thrust = 120000  # N
    b737_max_mass = 79015
    b737_dry_mass = 41500  # kg
    b737_fuel_mass = 26000  # kg
    b737_max_engine_mass = 2780  # kg
    jet_fuel_density = 0.804  # kg/L
    jet_fuel_energy_density = 43.15  # MJ/kg
    b737_turbine_power = b737_fuel_mass * jet_fuel_energy_density * req1_speed / b737_max_range  # kW

    # Convert cruising speed to m/s
    cruising_speed_m_s = (req1_speed * 1000) / 3600

    # Calculate range using the modified Breguet Range Equation for electric aircraft
    # Convert units
    battery_specific_energy_J_kg = battery_specific_energy_Wh_kg * 3600 * battery_multiplier # Convert Wh/kg to J/kg

    # Calculate heat management mass
    # split thrust by number of engines
    subreq1_motor_thrust = subreq1_motor_thrust / (num_engines)  # in N
    motor_power = subreq1_motor_thrust * cruising_speed_m_s * 1000  # in W
    motor_heat = (1 - eta_m) * subreq1_motor_thrust * req1_speed * (1000/3600) * 1000  # in W
    heat_mgmt_mass = motor_heat / (heat_dissipation_capacity * 1000)  # in kg
    total_motor_mass = subreq2_motor_mass + heat_mgmt_mass

    # Update total eTurbofan mass based on heat management mass
    total_mass_with_heat_mgmt = b737_max_engine_mass * num_engines + heat_mgmt_mass + battery_mass_kg + b737_dry_mass + req3_payload

    # Calculate range using the modified Breguet Range Equation for electric aircraft
    R_elec_m = (battery_specific_energy_J_kg / gravitational_constant) * lift_to_drag_ratio * (battery_mass_kg / total_mass_with_heat_mgmt) * eta_i * eta_m * eta_p
    R_elec_km = R_elec_m / 1000  # Convert to km

    # Compare with Boeing 737-800 MAX
    percentage_difference_range = ((R_elec_km / b737_max_range) - 1) * 100  # percentage difference
    percentage_difference_heat_mgmt_mass = ((heat_mgmt_mass / b737_max_engine_mass) - 1) * 100  # percentage difference

    # Display main numbers as metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Calculated e737 Range (km)", f"{R_elec_km:.2f}", f"{percentage_difference_range:.2f}% of 737-800 MAX")
    col1.metric("Heat Management Mass (kg)", f"{heat_mgmt_mass:.2f}", f"{percentage_difference_heat_mgmt_mass:.2f}% of 737-800 MAX engine mass", delta_color="inverse")
    col2.metric("Target eTurbine Thrust Required (kN)", f"{subreq1_motor_thrust:.2f}")
    col2.metric("eTurbine Power Required (kW)", f"{motor_power / 1000:.2f}")
    col2.metric("Motor Heat (kW)", f"{motor_heat / 1000:.2f}")
    col3.metric("eTurbofan Mass (kg)", f"{total_motor_mass:.2f}", f"{(total_motor_mass / b737_max_engine_mass - 1) * 100:.2f}% of 737-800 MAX engine mass")
    col3.metric("e737 Mass (kg)", f"{total_mass_with_heat_mgmt:.2f}", f"{(total_mass_with_heat_mgmt / b737_max_mass - 1) * 100:.2f}% of 737-800 MAX")

    # make a plotly figure that shows how range changes with values for battery mass and battery specific energy
    # generate a grid of values for battery mass and battery specific energy
    battery_masses = np.linspace(1000, 50000, 100)
    battery_specific_energies = np.linspace(100, 500, 100)
    battery_masses_grid, battery_specific_energies_grid = np.meshgrid(battery_masses, battery_specific_energies)
    # calculate range for each value in the grid
    R_elec_m_grid = (battery_specific_energies_grid * 3600 * battery_multiplier / gravitational_constant) * lift_to_drag_ratio * (battery_masses_grid / total_mass_with_heat_mgmt) * eta_i * eta_m * eta_p
    R_elec_km_grid = R_elec_m_grid / 1000  # Convert to km

    # create a plotly figure with two subplots side by side
    fig2 = make_subplots(rows=1, cols=1)

    # add a contour plot for range vs battery mass to the first subplot
    fig2.add_trace(go.Contour(x=battery_masses, y=battery_specific_energies, z=R_elec_km_grid, colorscale='Plasma', showscale=False), row=1, col=1)

    # add a scatter plot for the control point to both subplots
    fig2.add_trace(go.Scatter(
                                x=[battery_mass_kg],
                                y=[battery_specific_energy_Wh_kg],
                                name='Design',
                                hovertext=f'Range: {R_elec_km:.2f} km',
                                mode='markers',
                                marker=dict(color='white',size=10)),
                                row=1,
                                col=1
                                )

    # update the layout
    fig2.update_layout(
        title="Range vs Battery Mass and Battery Specific Energy",
        xaxis_title="Battery Mass (kg)",
        yaxis_title="Battery Specific Energy (Wh/kg)",
        height=500,
        width=1000,
        # make legend horizontal
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="right", x=1),
        # make hover value lable say Battery Specific Energy (x), Battery Mass (y), and Range (z)
        hoverlabel=dict(
            font_size=11,
            font_family="Rockwell"
        ),
        hovermode="closest"
    )

    # display the plotly figure
    st.plotly_chart(fig2, use_container_width=True)

    # make a plotly contour plot that shows how the motor and propulsive efficiency affect range
    # generate a grid of values for motor and propulsive efficiency
    motor_efficiencies = np.linspace(0.5, 1.0, 100)
    propulsive_efficiencies = np.linspace(0.5, 1.0, 100)
    motor_efficiencies_grid, propulsive_efficiencies_grid = np.meshgrid(motor_efficiencies, propulsive_efficiencies)
    # calculate range for each value in the grid
    R_elec_m_grid = (battery_specific_energy_J_kg / gravitational_constant) * lift_to_drag_ratio * (battery_mass_kg / total_mass_with_heat_mgmt) * eta_i * motor_efficiencies_grid * propulsive_efficiencies_grid
    R_elec_km_grid = R_elec_m_grid / 1000  # Convert to km

    # create a plotly figure with two subplots side by side (one for range vs motor efficiency vs propulsive efficiency and one for range vs inverter efficiency vs propulsive efficienc)
    fig3 = make_subplots(rows=1, cols=2, subplot_titles=("Motor Efficiency vs Propulsive Efficiency", "Inverter Efficiency vs Propulsive Efficiency"))

    # add a contour plot for range vs motor and propulsive efficiency to the first subplot
    fig3.add_trace(go.Contour(x=motor_efficiencies, y=propulsive_efficiencies, z=R_elec_km_grid, colorscale='Plasma', showscale=False), row=1, col=1)

    # add a scatter plot for the control point to both subplots
    fig3.add_trace(go.Scatter(
                                x=[eta_m],
                                y=[eta_p],
                                name='Design',
                                hovertext=f'Range: {R_elec_km:.2f} km',
                                mode='markers',
                                marker=dict(color='white',size=10)),
                                row=1,
                                col=1
                                )
    
    # add a contour plot for range vs inverter and propulsive efficiency to the second subplot
    fig3.add_trace(go.Contour(x=motor_efficiencies, y=propulsive_efficiencies, z=R_elec_km_grid, colorscale='Plasma', showscale=False), row=1, col=2)

    # add a scatter plot for the control point to both subplots
    fig3.add_trace(go.Scatter(
                                x=[eta_i],
                                y=[eta_p],
                                name='Design',
                                hovertext=f'Range: {R_elec_km:.2f} km',
                                mode='markers',
                                marker=dict(color='white',size=10)),
                                row=1,
                                col=2
                                )
    
    # update the layout
    fig3.update_layout(
        title="Range vs Motor and Propulsive Efficiency",
        xaxis_title="Motor Efficiency",
        yaxis_title="Propulsive Efficiency",
        height=500,
        width=1000,
        # make legend horizontal
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="right", x=1),
        # make hover value lable say Motor Efficiency (x), Propulsive Efficiency (y), and Range (z)
        hoverlabel=dict(
            font_size=11,
            font_family="Rockwell"
        ),
        hovermode="closest"
    )

    # display the plotly figure
    st.plotly_chart(fig3, use_container_width=True)