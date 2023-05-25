import pandas as pd
import numpy as np
import openai
import os

df_val = pd.read_csv('datos procesados/valores.csv')
dfi = pd.read_csv('datos procesados/componentes.csv', low_memory=False)

# Function to determine the limit surpassed and its value
def find_limit_surpassed(row):
    """
    Determines the limit surpassed and its corresponding value based on the given row.

    Parameters:
        row (pandas.Series): A row of a DataFrame containing the necessary columns.

    Returns:
        tuple: A tuple containing the name of the limit surpassed and its value.
    """
    # Check if the value is less than or equal to the lower critical limit
    if row['valor'] <= row['limite inferior condenatorio']:
        return 'limite inferior condenatorio', row['limite inferior condenatorio']
    # Check if the value is greater than or equal to the upper critical limit
    elif row['valor'] >= row['limite superior condenatorio']:
        return 'limite superior condenatorio', row['limite superior condenatorio']
    # Check if the value is less than or equal to the lower marginal limit
    elif row['valor'] <= row['limite inferior marginal']:
        return 'limite inferior marginal', row['limite inferior marginal']
    # Check if the value is greater than or equal to the upper marginal limit
    elif row['valor'] >= row['limite superior marginal']:
        return 'limite superior marginal', row['limite superior marginal']
    # If no limit is surpassed, return appropriate values
    else:
        return 'No limit surpassed', None
    
def value_prompt(df_in):
    """
    Generates a prompt string with the element, value, limit surpassed, and limit value.

    Parameters:
        df_in (pandas.DataFrame): The input DataFrame containing the necessary columns.

    Returns:
        str: A string representation of the prompt with the element, value, limit surpassed, and limit value.

    """
    # Create a copy of the input DataFrame
    df = df_in.copy()
    # Apply the 'find_limit_surpassed' function to create new columns
    df[['limite transgredido', 'valor limite']] = df.apply(find_limit_surpassed, axis=1, result_type='expand')
    # Select desired columns in the DataFrame
    df = df[['elemento', 'valor', 'limite transgredido', 'valor limite']]
    # Convert the DataFrame to a string representation
    prompt = df.to_string(index=False)

    return prompt

# Function to generate the prompt to chatgpt
def generate_prompt(gen, val):
    """
    Generates the main prompt for comments based on GPT-3.

    Args:
        row (pd.Series): Series representing a row from the table, containing the component information.
        val (pd.DataFrame): DataFrame containing the values of the sample.

    Returns:
        str: Generated prompt for GPT-3 chat interaction.
    """
    # Execute functions to get characteristics of sample
    com_prompt = component_prompt(gen)  # Call component_prompt() function to generate the component prompt for the sample
    val_prompt = value_prompt(val)  # Convert the DataFrame 'val' to a string representation

    # Unify prompts
    prompt = 'Analiza una muestra para el siguiente equipo:\n' + com_prompt + '\n' + 'Los valores de la muestra son:\n' + val_prompt

    return prompt

def component_prompt(row):
    """
    Generates the prompt that extract the info from component table using GPT-3.

    Args:
        row (pd.Series): Series representing a row from the table.

    Returns:
        str: Generated component prompt for GPT-3 interaction.
    """

    # Add context for the interaction with GPT-3
    context = {"role": "system",
                "content": "Eres un asistente que resume la información de una tabla a breves bullet points, evitando la redundancia"}
    messages = [context]

    # Set previous interaction to refine results
    ## Example 1
    messages.append({"role": "user", "content": """Analiza los componentes y características de una máquina identificada en la siguiente tabla:
                                            'name_component name_component_brand name_component_model name_component_type name_machine name_machine_type name_machine_brand name_machine_model\n  MOTOR DIESEL          CATERPILLAR                3516B        MOTOR DIESEL      CAEX 53      CAMIONES 793        CATERPILLAR               793B'
                                            """})
    messages.append({"role": "assistant", "content": "Componente: Motor Diesel Caterpillar 3516B\nMáquina: Camión Caterpillar 793B"})

    ## Example 2
    messages.append({"role": "user", "content": """Analiza los componentes y características de una máquina identificada en la siguiente tabla:
                                            'name_component name_component_brand name_component_model name_component_type name_machine name_machine_type name_machine_brand name_machine_model\n  GRASA AC-100                  NaN                  NaN            DEPOSITO CAMION GRASA            CAMION                NaN                NaN'
                                            """})
    messages.append({"role": "assistant", "content": "Componente: Deposito de grasa\nMáquina: Camión grasa"})

    ## Example 3
    messages.append({"role": "user", "content": """Analiza los componentes y características de una máquina identificada en la siguiente tabla:
                                            'name_component name_component_brand name_component_model name_component_type        name_machine name_machine_type name_machine_brand name_machine_model\n      REDUCTOR          NEU-SOCOFER                  NaN            REDUCTOR ASPIRADOS DE VIA 01  Aspirado de Vias        NEU-SOCOFER                NaN'
                                            """})
    messages.append({"role": "assistant", "content": "Componente: Reductor Neu-socofer\nMáquina: Aspirados de via Neu-socofer"})

    ## Example 4
    messages.append({"role": "user", "content": """Analiza los componentes y características de una máquina identificada en la siguiente tabla:
                                            'name_component name_component_brand name_component_model name_component_type name_machine     name_machine_type name_machine_brand name_machine_model\n      REDUCTOR                  NaN                  NaN            REDUCTOR         CT-2 CORREA TRANSPORTADORA                NaN                NaN'
                                            """})
    messages.append({"role": "assistant", "content": "Componente: Reductor\nMáquina: Correa Transportadora"})

    ## Example 5
    messages.append({"role": "user", "content": """Analiza los componentes y características de una máquina identificada en la siguiente tabla:
                                            'name_component name_component_brand name_component_model name_component_type name_machine      name_machine_type name_machine_brand name_machine_model\n  MOTOR DIESEL              CUMMINS                  NaN        MOTOR DIESEL     CAEX-726 CAMIONES DE EXTRACCIÓN            KOMATSU              730-E'
                                            """})
    messages.append({"role": "assistant", "content": "Componente: Motor Diesel Cummins\nMáquina: Camión Komatsu 730-E"})
    
    ## Example 6
    messages.append({"role": "user", "content": """Analiza los componentes y características de una máquina identificada en la siguiente tabla:
                                            '    name_component name_component_brand name_component_model name_component_type name_machine name_machine_type name_machine_brand name_machine_model\nACEITE MOTOR 15W40                  NaN                  NaN        MOTOR DIESEL  DESCONOCIDO       DESCONOCIDO                NaN                NaN'
                                            """})
    messages.append({"role": "assistant", "content": "Componente: Aceite motor Diesel 15W40\n"})

    ## Example 7
    messages.append({"role": "user", "content": """Analiza los componentes y características de una máquina identificada en la siguiente tabla:
                                            'name_component name_component_brand name_component_model name_component_type                   name_machine name_machine_type name_machine_brand name_machine_model\n  MOTOR DIESEL                  NaN                  NaN        MOTOR DIESEL CAMION TOLVA 11 M3 - 03-105610    CAMIONES TOLVA                NaN                NaN'
                                            """})
    messages.append({"role": "assistant", "content": "Componente: Motor Diesel\nMáquina: Camión Tolva"})

    ## Example 8
    messages.append({"role": "user", "content": """Analiza los componentes y características de una máquina identificada en la siguiente tabla:
                                            'name_component name_component_brand name_component_model name_component_type name_machine      name_machine_type name_machine_brand name_machine_model\n  MOTOR DIESEL              CUMMINS                  NaN        MOTOR DIESEL     CAEX-722 CAMIONES DE EXTRACCIÓN            KOMATSU              730-E'
                                            """})
    messages.append({"role": "assistant", "content": "Componente: Motor Diesel Cummins\nMáquina: Camión de Extracción 730-E Komatsu"})

    ## Example 9
    messages.append({"role": "user", "content": """Analiza los componentes y características de una máquina identificada en la siguiente tabla:
                                            'name_component name_component_brand name_component_model name_component_type       name_machine  name_machine_type name_machine_brand name_machine_model\n  MOTOR DIESEL                  NaN                  NaN        MOTOR DIESEL PERFORADORA D - 06 FLOTA PERFORADORAS        ATLAS COPCO                NaN'
                                            """})
    messages.append({"role": "assistant", "content": "Componente: Motor Diesel\nMáquina: Perforadora D-06 Atlas Copco"})

    # Add user message providing the specific row information
    messages.append({"role": "user", "content": f"""Analiza los componentes y características de una máquina identificada en la siguiente tabla:
                                            {row.to_string(index=False)}
                                            """})

    # Call OpenAI API to generate the response based on the conversation history
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return response.choices[0].message.content  # Return the content of the assistant's response

def excecute_prompt(prompt):
    """
    Executes a chat prompt with GPT-3 and retrieves the response.

    Args:
        prompt (str): Prompt for GPT-3 chat interaction.

    Returns:
        str: Generated response from GPT-3.
    """

    # Add context for the interaction with GPT-3
    context = {"role": "system",
                "content": "Eres un ingeniero mecanico, especialista en equipos mineros y debes realizar diagnosticos precisos sobre las medidas de un equipo, entregando comentarios breves respecto a los análisis de aceite realizados y recomendaciones concretas de mantención. Tus respuestas deben ser de 150 palabras o menos"}
    messages = [context]

    # Set previous interaction to refine results
    ## Example 1
    messages.append({"role": "user", "content": 'Analiza una muestra para el siguiente equipo:\nComponente: Aceite motor Diesel 15W40\nLos valores de la muestra son:\n                    elemento  valor          limite transgredido  valor limite\n           Contenido de agua    8.3 limite superior condenatorio           0.3\nViscosidad cinemática @ 40°C  144.6 limite superior condenatorio         138.0'})
    messages.append({"role": "assistant", "content": 'Se aprecian niveles de desgaste y contaminación externa entre límites permisibles, sin embargo, se detecta contenido de agua 8,3% en volumen de muestra. Grado de viscosidad sobre límite superior condenatorio 144,6 por posibles rellenos con lubricante de ISO VG mayor.\nSe sugiere cambio de lubricante y mantener seguimiento riguroso cada 125 horas, para evidenciar alzas abruptas de sodio y potasio por eventual traspaso de refrigerante. Monitorear consumos de aceite y gases blancos en salida de escape.'})
   
    ## Example 2
    messages.append({"role": "user", "content": 'Analiza una muestra para el siguiente equipo:\nComponente: Motor Diesel\nMáquina: Camión Tolva\nLos valores de la muestra son:\n                     elemento  valor          limite transgredido  valor limite\n                       Fierro   31.1     limite superior marginal          30.0\n                        Cobre  267.7 limite superior condenatorio          15.0\n                       Estaño    2.9     limite superior marginal           1.5\n                      Silicio   30.7 limite superior condenatorio          17.0\n                    Índice PQ   20.0     limite superior marginal          20.0\nViscosidad cinemática @ 100°C   10.7 limite inferior condenatorio          11.0\n     Dilución por combustible    0.8 limite superior condenatorio           0.5'})
    messages.append({"role": "assistant", "content": 'Se detecta concentración de metales de desgaste por Fierro 31.1 ppm y Cobre 267.7 ppm, evidenciando posible abrasión excesiva en cojinetes y bujes de turbo. Silicio 30.7 ppm señala ingesta excesiva de polvo ambiental. Análisis fisicoquímico detecta material ferroso en suspensión Índice PQ 20.0. Se evidencia Dilución por combustible de 0.8%, detectando posible anormalidad en bomba de transferencia o sellos de inyectores, provocando una baja ligera en viscosidad en lubricante.\nSe recomienda priorizar cambio de lubricante y elementos filtrantes, evaluar presiones en sistema de lubricación y saturación temprana de filtros, mantener seguimiento riguroso cada 50 hrs.'})
    
    ## Example 3
    messages.append({"role": "user", "content": 'Analiza una muestra para el siguiente equipo:\nComponente: Motor Diesel Cummins\nMáquina: Camión Komatsu 730-E de Extracción\nLos valores de la muestra son:\n                elemento  valor          limite transgredido  valor limite\nDilución por combustible    2.2 limite superior condenatorio           0.3'})
    messages.append({"role": "assistant", "content": 'Niveles de desgaste y contaminaciones determinadas por espectrometría en tamaño y concentraciones admisibles para la operación, sin embargo, análisis físico químico detecta Dilución por combustible 2.2%, altamente perjudicial para la operación, evidenciando posibles anomalías en toberas o sellos de inyectores. Grado de visosidad normal en lubricante.\nSe sugiere priorizar intervención mecánica y efectuar cambio de lubricante, junto con envío de contramuestra para realizar seguimiento a deterioro en sellos/toberas de inyectores o bomba de transferencia. Evaluar presiones en sistema de lubricación y saturación temprana de filtros.'})    

    # Add user message containing the prompt
    messages.append({"role": "user", "content": prompt})

    # Call OpenAI API to generate the response based on the conversation history
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return response.choices[0].message.content  # Return the content of the assistant's response



def generate_comment(id, df_general=dfi, df_values=df_val):
    """
    Generates a comment based on the sample id and its associated data.

    Args:
        id (int): The sample id for which to generate the comment.
        df_general (pandas.DataFrame, optional): The DataFrame containing general information about the samples.
            Defaults to dfi.
        df_values (pandas.DataFrame, optional): The DataFrame containing the values associated with the samples.
            Defaults to df_val.

    Returns:
        tuple: A tuple containing the generated response comment and the corresponding prompt.

    Raises:
        None

    Notes:
        - This function assumes that the 'df_general' DataFrame has the columns 'id_sample', 'name_component',
        'name_component_type', 'name_machine', and 'name_machine_type'.
        - This function assumes that the 'df_values' DataFrame has the columns 'id_sample' and any other necessary
        columns containing the values of interest.

    """

    # Get the sample rows
    ## General row
    gen = df_general[df_general.id_sample == id][['name_component','name_component_brand','name_component_model','name_component_type','name_machine','name_machine_type','name_machine_brand','name_machine_model']]
    ## Values rows
    val = df_val[df_val.id_sample == id].drop(columns=['id_sample'])

    # If no limit is surpassed
    if val.shape[0] == 0:
        response = "Niveles de desgaste junto a contaminación externa dentro de los límites permisibles para el servicio.\nGrado de viscocidad y degradación normal de lubricante.\nContinuar con monitoreo de lubricante y componente según plan de mantenimiento."""
        prompt = ''

    #If any limit is surpassed
    else:
        prompt = generate_prompt(gen, val)
        response = excecute_prompt(prompt)
    # response = ''

    return (response, prompt)