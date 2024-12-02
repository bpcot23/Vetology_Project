import json
import os
import pandas as pd
import re


from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser




def load_config(config_path):
    """
    Load config from a given path.
    
    Args:
        config_path (str): The path to the config file.

    Returns:
        dict: The loaded config if successful, otherwise None.
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


class DataProcessor:

    def __init__(self, datapath, conditions):
        """
        Initialize model variables referenced throughout class functions

        Args:
            datapath (str): A filepath to the .csv file containing radiology report data
            conditions (List (str)): A list of possible conditions
        """
        self.datapath = datapath
        self.raw_data = None
        self.cleaned_data = None
        self.conditions = conditions


    def data_to_df(self):
        """
        Uses path to the dataset excel .csv file, reads file into a pandas dataframe

        Returns:
            None
        """
        raw_data = pd.read_csv(self.datapath, sep = ",", encoding = "utf-8-sig")
        self.raw_data = raw_data


    def clean_and_group_data(self):
        """
        Cleans the data by standardizing whitespace, adds new columns that contain the full radiologist report, the full AI report,
        and a compiled list of AI decisions (normal/abnormal) on specfic conditions. 

        Returns:
            None
        """
        raw_data = self.raw_data

        # Standardizes whitespace across dataset
        for col in raw_data.columns:
            if raw_data[col].dtype == 'object':
                raw_data[col] = raw_data[col].fillna('') 
                raw_data[col] = raw_data[col].apply(lambda x: re.sub(r'(\s|<br>)+', ' ', str(x)))
        
        # Add new columns to dataframe
        raw_data.insert(3, "Full Report (original radiologist report)", "", True)
        raw_data.insert(4, "Full Report (AI report)", "", True)
        raw_data.insert(5, "Conditions (AI report)", "", True)
        raw_data.insert(6, "Difference Assessment", "", True)


        for index, row in raw_data.iterrows():
            print("Cleaning row #" + str(index))

            # Combines radiologist report findings, conclusions, and recommendations into a single report
            findings = row["Findings (original radiologist report)"]
            conclusions = row["Conclusions (original radiologist report)"]
            recommendations = row["Recommendations (original radiologist report)"]

            report = "findings:\n" + findings + "\n"

            # Conclusions/Recommendations in separate columns in some reports: label accordingly
            if recommendations != "As above.":
                report = report + "conclusions:\n" + conclusions + "\n"
                report = report + "recommendations:\n" + recommendations

            # Conclusions/Recommendations grouped together in some reports: label accordingly
            else:
                report = report + "conclusions and recommendations:\n" + conclusions

            # Update dataframe
            raw_data.at[index, "Full Report (original radiologist report)"] = report


            # Combines AI report findings, conclusions, and recommendations into a single report
            ai_findings = row["Findings (AI report)"]
            ai_conclusions = row["Conclusions (AI report)"]
            ai_recommendations = row["Recommendations (AI report)"]

            report = "findings:\n" + ai_findings + "\n"

            # Conclusions/Recommendations in separate columns in some reports: label accordingly
            if recommendations != "":
                report = report + "conclusions:\n" + ai_conclusions + "\n"
                report = report + "recommendations:\n" + ai_recommendations

            # Recommendations not included in some reports: label accordingly
            else:
                report = report + "conclusions:\n" + ai_conclusions

            # Update dataframe
            raw_data.at[index, "Full Report (AI report)"] = report


            # Compile AI decisions on specific conditions into single string
            Comparison_conditions = ""
            for condition in self.conditions:
                result = row[condition]

                # vhs_v2 is labelled differently in the Excel sheet, this must be corrected
                if condition == 'vhs_v2':
                    if result == 'enlarged':
                        result == 'abnormal'
                    elif result == 'normal':
                        pass
                    else:
                        result == 'normal'
                Comparison_conditions = Comparison_conditions + condition + ": " + result + "\n"

            # Update dataframe
            raw_data.at[index, "Conditions (AI report)"] = Comparison_conditions

        # Save cleaned dataframe
        self.cleaned_data = raw_data

    def clean_gold_data(self, gold_datapath):
        """
        Function to process and group the gold standard data before analysis.

        Args:
            gold_datapath (str): the filepath to the gold standard classifications

        Returns:
            gold_data (dataframe): An updated file containing all the gold standard classifications
        """

        gold_data = pd.read_csv(gold_datapath, sep = ",", encoding = "latin1")
        gold_data.insert(1, "Gold Conditions", "", True)

        for index, row in gold_data.iterrows():

        # Compile gold decisions on specific conditions into single string
            Comparison_conditions = ""
            for condition in self.conditions:
                result = row[condition]
                Comparison_conditions = Comparison_conditions + condition + ": " + result + "\n"

            # Update dataframe
            gold_data.at[index, "Gold Conditions"] = Comparison_conditions
        
        return gold_data




    def Update_Sensitivity(self, row):
        """
        Function returns the calculated sensitivity value for the condition
        """
        try:
            output = row['tp_Positive'] / (row['tp_Positive'] + row['fn_Positive'])
        except ZeroDivisionError:
            output = 0
        return output

    def Update_Specificity(self, row):
        """
        Function returns the calculated specificity value for the condition
        """
        try:
            output = row['tn_Positive'] / (row['tn_Positive'] + row['fp_Positive'])
        except ZeroDivisionError:
            output = 0
        return output

    def Update_Check(self, row):
        """
        Function returns the sum of the four classification groups
        """
        return row['tp_Positive'] + row['fn_Positive'] + row['tn_Positive'] + row['fp_Positive']
    
    def Update_GT_Check(self, row):
        """
        Function returns the sum of the Positive and Negative Ground Truth Numbers
        """
        return row['Positive Ground Truth'] + row['Negative Ground Truth']

    def Update_Accuracy(self, row):
        """
        Function returns the percentage of cases correctly classified by the model
        """
        return (row['tn_Positive'] + row['tp_Positive']) / (row['tn_Positive'] + row['tp_Positive'] + row['fn_Positive'] + row['fp_Positive'])
    def Confusion_Matrix(self, data, num_rows, condition_list, matrix_path):
        """
        Function creates a confusion matrix based on the data generated through the model.

        Args:
            data (pd.DataFrame): the dataset resulting from the model.
            num_rows (int): the number of reports that were processed by the model, used to control
                            how many rows from the dataframe are included in the confusion matrix.
            condition_list (List (str)): list of all conditions the model classified.
            matrix_path (str): path to the file location where the confusion matrix should be saved.

        Returns:
            None
        """

        matrix = pd.DataFrame(0,
                              columns=['condition', 'tp_Positive', 'fn_Positive', 
                                       'tn_Positive', 'fp_Positive', 'Sensitivity', 
                                       'Specificity', 'Accuracy', 'Check', 'Positive Ground Truth', 
                                       'Negative Ground Truth', 'Ground Truth Check'],
                              index = condition_list
                            )

        matrix['condition'] = condition_list

        # Iterate over each frow in dataframe
        for index, row in data.iterrows():

            # Specifies which row in df to stop at
            if index == num_rows:
                break

            else:
                # Gather all report decisions in a list
                report = row["Difference Assessment"]
                cond_decisions = report.split('\n')

                for line in cond_decisions:

                    # Clean each line to isolate the condition name and decision made by the model
                    condition_match = re.search(r'^[\w\s\-]+:', line)

                    if condition_match:
                        condition_raw = condition_match.group().strip()
                        condition_raw = re.sub(" ", "_", condition_raw)
                    else:
                        print(line)
                        continue
                    condition = re.sub(":", "", condition_raw)

                    # Address label split by '_' inside word boundary
                    if condition == "hypoplastic_trachea":
                        condition = "hypo_plastic_trachea"

                    decision_match = re.search(r'(True|False) (Positive|Negative)', line)
                    
                    decision = decision_match.group()

                    if condition in condition_list:

                        # Update columns based on decision value
                        if decision == "True Negative":
                            matrix.loc[condition, "tn_Positive"] += 1
                            matrix.loc[condition, 'Negative Ground Truth'] += 1

                        elif decision == "True Positive":
                            matrix.loc[condition, 'tp_Positive'] += 1
                            matrix.loc[condition, 'Positive Ground Truth'] += 1

                        elif decision == "False Negative":
                            matrix.loc[condition, 'fn_Positive'] += 1
                            matrix.loc[condition, 'Positive Ground Truth'] += 1

                        elif decision == "False Positive":    
                            matrix.loc[condition, 'fp_Positive'] += 1
                            matrix.loc[condition, 'Negative Ground Truth'] += 1
                    else:
                        continue
        
        # Update remaining columns given total counts
        matrix['Sensitivity'] = matrix.apply(self.Update_Sensitivity, axis=1)
        matrix['Specificity'] = matrix.apply(self.Update_Specificity, axis=1)
        matrix['Accuracy'] = matrix.apply(self.Update_Accuracy, axis=1)
        matrix['Check'] = matrix.apply(self.Update_Check, axis=1)
        matrix['Ground Truth Check'] = matrix.apply(self.Update_GT_Check, axis=1)

        matrix.to_csv(matrix_path)



class Agents:

    def __init__(self, model, conditions):
        """
        Initialize model variables referenced throughout class functions

        Args:
            model (str): the name of the model used for the Agents
            conditions (list(str)): A list of the conditions for classification
        """
        if model == "gpt-4o-mini":
            self.Actor = ChatOpenAI(model_name = model, temperature = 0)
            self.Evaluator = ChatOpenAI(model_name = model, temperature = 0)
            self.Self_Reflection = ChatOpenAI(model_name = model, temperature = 0)
            self.ConditionID = ChatOpenAI(model_name = model, temperature = 0)
        elif model == "claude-3-haiku-20240307":
            self.Actor = ChatAnthropic(model=model,
                                    temperature=0,
                                    max_tokens=1024,
                                    timeout=None,
                                    max_retries=2,
                                    )
            self.Evaluator = ChatAnthropic(model=model,
                                    temperature=0,
                                    max_tokens=1024,
                                    timeout=None,
                                    max_retries=2,
                                    )
            self.Self_Reflection = ChatAnthropic(model=model,
                                    temperature=0,
                                    max_tokens=1024,
                                    timeout=None,
                                    max_retries=2,
                                    )
            self.ConditionID = ChatAnthropic(model=model,
                                    temperature=0,
                                    max_tokens=1024,
                                    timeout=None,
                                    max_retries=2,
                                    )
        elif model == "llama3-8b":
            self.Actor = ChatBedrock(model = model)
            self.Evaluator = ChatBedrock(model = model)
            self.Self_Reflection = ChatBedrock(model = model)
            self.ConditionID = ChatBedrock(model = model)
        else:
            raise ValueError("Unexpected model name. Please try again")
        self.parser = StrOutputParser()
        self.conditions = conditions


    def process_Reflexion_prompts(self, actor_path, evaluator_path, selfreflection_path):
        """
        This function reads in the prompts that will be used by the model agents, and establishes prompt
        templates for each prompt that allows them to be revised throughout the reinforcement learning process

        Args:
            actor_path (str): the filepath to the prompt for the Actor agent
            evaluator_path(str): the filepath to the prompt for the Evaluator agent
            selfreflection_path (str): the filepath to the prompt for the Self Reflection agent

        Returns:
            None
        """

        with open(actor_path, "r") as a:
            A_prompt_sys = a.read()

            A_prompt_hum = PromptTemplate(
                input_variables = ['action_1', 'report'],
                template = """
                Additional instructions:
                {action_1}
                These instructions must be followed exactly when performing your classification task. 

                Report:
                {report}
                """
            )

        with open(evaluator_path, "r") as e:
            E_prompt_sys = e.read()

            E_prompt_hum = PromptTemplate(
                input_variables = ['report', 'actor_decision'],
                template = """
                Report:
                {report}

                Teammate 1 Condition Classification Decisions:
                {actor_decision}
                """
            )

        with open(selfreflection_path, "r") as sr:
            SR_prompt_sys = sr.read()

            SR_prompt_hum = PromptTemplate(
                input_variables = ['report', 'actor_decision', 'evaluation', 'last_feedback'],
                template = """
                Report:
                {report}

                Teammate 1 Condition Classification Decisions:
                {actor_decision}

                Teammate 2 Evaluation:
                {evaluation}

                Previous Feedback:
                {last_feedback}
                """
            )

        self.A_prompt_sys = A_prompt_sys
        self.A_prompt_hum = A_prompt_hum
        self.E_prompt_sys = E_prompt_sys
        self.E_prompt_hum = E_prompt_hum
        self.SR_prompt_sys = SR_prompt_sys
        self.SR_prompt_hum = SR_prompt_hum


    def process_CoT_prompts(self, cond_path):
        """
        This function reads in the prompts that will be used by the model agents, and establishes prompt
        templates for each prompt that allows them to be revised based on the Chain of Thought process.

        Args:
            cond_path (str): the filepath to the prompt for the condition identification agent

        Returns:
            None
        """
        with open(cond_path, "r") as cond:
            cond_prompt_sys = cond.read()

            cond_prompt_hum = PromptTemplate(
                input_variables = ['report'],
                template = """
                Report:
                {report}
                """
            )    

        self.cond_prompt_sys = cond_prompt_sys
        self.cond_prompt_hum = cond_prompt_hum


    def Reflexion_model (self, radiologist_report, report2, gold_or_ai, ai_report_or_labels, ID,):
        """
        A function for managing calls to the Reflexion system.

        Args:
            radiologist_report (str): The full radiologist report (findings, conclusions, recommendations) on a radiology scan
            report2 (str): The second report. Could be AI labels or a report, or Gold standard classifications
            gold_or_ai (str): Indicates whether the report2 is gold labels or an AI report
            ai_report_or_labels (str): A variable describing the format of the AI report
            ID (str): The case ID of the given report, used to name file log
        
        returns:
            Assessment (str): The Classifier's assessment of differences between the two reports
        """
        # Use Reflexion to identify conditions in radiology report
        radiologist_assessment = self.Reflexion(radiologist_report)

        # Create Logs for Agent assessments, evaluation, and feedback
        filename = "logs/log_ID_" + ID + "_Rad.txt"
        with open(filename, "w") as f:
            f.write("Decision log:\n")
            for line in myAgents.full_decision_list:
                f.write(line + "\n\n")
            f.write("\nEvaluation log:\n")
            for line in myAgents.full_eval_list:
                f.write(line + "\n")
            f.write("\nFeedback log:\n")
            for line in myAgents.full_memory_list:
                f.write(line + "\n\n")
            f.write("Final Model Decision:\n")
            f.write(radiologist_assessment)

        # Path for AI report/label analysis
        if gold_or_ai == "ai":
            # Use Reflexion to identify conditions in radiology report
            if ai_report_or_labels == "report":
                
                assessment2 = self.Reflexion(report2)

                # Create Logs for Agent assessments, evaluation, and feedback
                filename = "logs/log_ID_" + ID + "_AI.txt"
                with open(filename, "w") as f:
                    f.write("Decision log:\n")
                    for line in myAgents.full_decision_list:
                        f.write(line + "\n\n")
                    f.write("\nEvaluation log:\n")
                    for line in myAgents.full_eval_list:
                        f.write(line + "\n\n")
                    f.write("\nFeedback log:\n")
                    for line in myAgents.full_memory_list:
                        f.write(line + "\n\n")

            # Reflexion not needed if condition labels already classified            
            else:
                assessment2 = report2

        # Path for gold label analysis
        else:
            assessment2 = report2

        # Initialize the prompt for classifying report conditions as TP, TN, FP, FN
        # If AI, we want to set the "gold" standard to the radiologist assessment
        # If Gold Labels, we set the "gold" standard to the gold labels
        if gold_or_ai == "ai":
            assessment = self.classify_conditions(report_a = radiologist_assessment, 
                                                  report_b = assessment2, 
                                                  conditions = self.conditions)
        else:
            assessment = self.classify_conditions(report_a = assessment2,
                                                  report_b = radiologist_assessment,
                                                  conditions = self.conditions)

        return assessment

        

    def Reflexion(self, radiologist_report):
        """
        This function uses reinforcement learning techniques based on 
        Shinn et al. (2023) "Reflexion: Language Agents with Verbal Reinforcement Learning"
        In this function, there are three agents: Actor, Evaluator, Self Reflection
            Actor is given a prompt to identify conditions in the report
            Evaluator is to identify all the Actor condition classifications which it disagrees with
            Self Reflection is given a prompt to provide the Actor with feedback on how to score better for the Evaluator

        These three agents take part in a reinforcement learning loop in which the Actor produces an assessment, the Evaluator
        scores that assessment, and the Self Reflection provides feedback for the Actor that is then used by the Actor in
        the following loop. This process stops when the Evaluator scores the Actor's assessment high enough, or the maximum 
        number of loop iterations is reached.

        Args:
            radiologist_report (str): A report (findings, conclusions, recommendations) on a radiology scan

        Returns:
            best_assessment (str): The Agent assessment of a report.
        """

        # Initialize storage variables, used as a log to check model actions
        self.full_decision_list = []
        self.full_memory_list = []
        self.full_eval_list = []

        # Initialize n, representing how many times the model has looped through
        n = 0
        # Initialize m, representing the maximum number of times the model can loop through
        m = 3
        # Initialize decision_diffs, a score of how many decisions the Evaluator disagrees with the Agent on 
        # If E_score is at or below decision_diffs, Actor performance is considered satisfactory
        decision_diffs = 0
        # Initialize A_prompt and action0
        formatted_A_prompt = self.A_prompt_hum.format(
            action_1 = "",
            report = radiologist_report,
        )

        A_messages = [
            SystemMessage(content=self.A_prompt_sys),
            HumanMessage(content=formatted_A_prompt)
        ]
        # Generate initial decision/trajectory: run Actor with A_prompt
        # Output: d_0, the initial model decision on whether two reports align or not, and where they disagree
        print("Generating initial report assessment...")
        A_result = self.Actor.invoke(A_messages)
        d_0 = self.parser.invoke(A_result)

        # Initialize E_prompt
        formatted_E_prompt = self.E_prompt_hum.format(
            report = radiologist_report,
            actor_decision = d_0
        )

        E_messages = [
            SystemMessage(content=self.E_prompt_sys),
            HumanMessage(content=formatted_E_prompt)
        ]
        # Evaluate initial decision: run Evaluator with E_prompt and d_0
        # Output: Evaluation, a list of all the condition classifications the Evaluator disagrees with
        print("Performing initial evaluation of assessment...")
        E_result = self.Evaluator.invoke(E_messages)
        Evaluation = self.parser.invoke(E_result)

        E_list = Evaluation.split()
        E_score = E_list[0]

        # Initialize best_score as E_score, will update if Reflexion improves score
        best_score = E_score
        best_assessment = d_0

        # Initialize SR_prompt
        formatted_SR_prompt = self.SR_prompt_hum.format(
            report = radiologist_report,
            actor_decision = d_0,
            evaluation = Evaluation,
            last_feedback = ""
        )
        SR_messages = [
            SystemMessage(content=self.SR_prompt_sys),
            HumanMessage(content=formatted_SR_prompt)
        ]

        # Generate self-reflection: run Self_Reflection with SR_prompt, Evaluation, and d_0
        # Output: action_1, the initial textual feedback for improving A_prompt
        print("Generating feedback...")
        SR_result = self.Self_Reflection.invoke(SR_messages)
        action_1 = self.parser.invoke(SR_result)

        # Initialize memory, initially consisting of [action_1]. This will be expanded with further iterations
        memory = [action_1]

        # Update logs
        self.full_decision_list.append(d_0)
        self.full_memory_list.append(action_1)
        self.full_eval_list.append(Evaluation)

        d_n = d_0

        # Create while loop for self-reinforcement
        while int(E_score) > decision_diffs and n < m:
            print("Entering reinforcement learning loop...")

            # Assign action_n variable based on memory
            action_1 = memory[-1]

            # Generate decision: run Actor with A_prompt, memory, and radiology reports
            # Output: d_n, the model decision at loop n
            formatted_A_prompt = self.A_prompt_hum.format(
                action_1 = action_1,
                report = radiologist_report,
            )
            A_messages = [
                SystemMessage(content=self.A_prompt_sys),
                HumanMessage(content=formatted_A_prompt)
            ]
            print("Generating updated assessment...")
            A_result = self.Actor.invoke(A_messages)
            d_n = self.parser.invoke(A_result)
            

            # Evaluate dn: run Evaluator with E_prompt and d_n
            # Output: Evaluation
            formatted_E_prompt = self.E_prompt_hum.format(
                report = radiologist_report,
                actor_decision = d_n
            )
            E_messages = [
                SystemMessage(content=self.E_prompt_sys),
                HumanMessage(content=formatted_E_prompt)
            ]
            print("Evaluating updated assessment...")
            E_result = self.Evaluator.invoke(E_messages)
            Evaluation = self.parser.invoke(E_result)

            E_list = Evaluation.split()
            E_score = E_list[0]

            # Update best_score and best_assessment if current E_score is lower/equal to its previous best
            if E_score <= best_score:
                best_score = E_score
                best_assessment = d_n

            # Determine if E_score is high enough to break from reinforcement learning loop
            if int(E_score) <= decision_diffs:

                # Update logs
                self.full_decision_list.append(d_n)
                self.full_eval_list.append(Evaluation)

                # Return d_n as the correct model decision
                return d_n

            else:

                # Generate self-reflection: run Self_Reflection with SR_prompt, evaluation, d_n, and action_n
                # Output: action_n
                formatted_SR_prompt = self.SR_prompt_hum.format(
                    report = radiologist_report,
                    actor_decision = d_n,
                    evaluation = Evaluation,
                    last_feedback = memory[-1]
                )
                SR_messages = [
                    SystemMessage(content=self.SR_prompt_sys),
                    HumanMessage(content=formatted_SR_prompt)
                ]
                print("Providing additional feedback...")
                SR_result = self.Self_Reflection.invoke(SR_messages)
                action_n = self.parser.invoke(SR_result)

                # Update memory to include action_n
                memory.append(action_n)
                
                # Update logs
                self.full_decision_list.append(d_n)
                self.full_eval_list.append(Evaluation)
                self.full_memory_list.append(action_n)

                # Update loop number
                n += 1
        
        # Return best_assessment as the model decision at end of while loop
        # This ensures that even if Reflexion model produced worse results over time,
        # The best outcome will be preserved and returned
        return best_assessment
            
    def ChainofThought(self, radiologist_report, gold_or_ai, report2, ai_report_or_labels):
        """
        This function utilizes multiple calls to LLM agents in order to break the task into smaller subtasks
        and improve reasoning capabilities. This function handles the classification of medical conditions based 
        on radiologist reports, and compares the outputs from two separate models (radiologist and ai/gold).

        Args:
            radiologist_report (str): The full radiologist report (findings, conclusions, recommendations) on a radiology scan
            report2 (str): The second report. Could be AI labels or a report, or Gold standard classifications
            gold_or_ai (str): Indicates whether the report2 is gold labels or an AI report
            ai_report_or_labels (str): A variable describing the format of the AI report
        
        returns:
            assessment (str): The Classifier's assessment of differences between the two reports
        """

        # Format condition identification prompt for Original Radiologist report
        formatted_cond_prompt = self.cond_prompt_hum.format(
            report = radiologist_report
        )

        cond_messages = [
            SystemMessage(content=self.cond_prompt_sys),
            HumanMessage(content=formatted_cond_prompt)
        ]

        # Generate list of condition identifications in Original Radiologist report
        print("Generating radiologist conditions list...")
        cond_result = self.ConditionID.invoke(cond_messages)
        radiologist_conditions = self.parser.invoke(cond_result)

        if gold_or_ai == "ai":
            # Create conditions list if 'report', read in list if 'labels'
            if ai_report_or_labels == "report":

                # Format condition identification prompt for AI report
                formatted_cond_prompt = self.cond_prompt_hum.format(
                    report = report2
                )

                cond_messages = [
                    SystemMessage(content=self.cond_prompt_sys),
                    HumanMessage(content=formatted_cond_prompt)
                ]
                # Generate list of differences between Reports A and B
                print("Generating AI conditions list...")
                cond_result = self.ConditionID.invoke(cond_messages)
                assessment2 = self.parser.invoke(cond_result)
            
            else:
                print("Generating AI conditions list...")

                # set equal to report2 if report is already a list of labels
                assessment2 = report2
        
        else:
            assessment2 = report2

        # Initialize the prompt for classifying report conditions as TP, TN, FP, FN
        # If AI, we want to set the "gold" standard to the radiologist assessment
        # If Gold Labels, we set the "gold" standard to the gold labels
        if gold_or_ai == "ai":
            assessment = self.classify_conditions(report_a = radiologist_conditions, 
                                                  report_b = assessment2, 
                                                  conditions = self.conditions)
        else:
            assessment = self.classify_conditions(report_a = assessment2,
                                                  report_b = radiologist_conditions,
                                                  conditions = self.conditions)

        return assessment

    def classify_conditions(self, report_a, report_b, conditions):
        """
        This function handles the classification of medical conditions based on comparisons between two reports
        and assigns classifications to conditions according to specified rules.

        Args:
            report_a_str (str): The first report (considered the Gold standard) with conditions and their labels.
            report_b_str (str): The second report to compare, such as a model or AI-generated report.
            conditions (list of str): A list of conditions to classify based on comparisons between the two reports.

        Returns:
            list of str: A list of classification results comparing the conditions from both reports, formatted
                        as strings that describe the classification and the Gold and Model labels.
        """
        # Parse each report into dictionaries
        def parse_report(report_str):
            report_dict = {}
            for line in report_str.strip().split("\n"):
                try:
                    condition, label = line.split(": ")
                    condition = re.sub(r' ', '_', condition)
                    if condition == "hypoplastic_trachea":
                        condition = "hypo_plastic_trachea"
                    report_dict[condition.strip()] = label.strip()
                except ValueError:
                    pass
            return report_dict
        
        print(report_a)
        report_a = parse_report(report_a)
        print(report_b)
        report_b = parse_report(report_b)

        results = []
        for condition in conditions:
            # Get labels for each condition; default to 'normal' if missing
            gold_label = report_a.get(condition, 'normal')
            model_label = report_b.get(condition, 'normal')
            
            # Classify based on specified rules
            if gold_label == 'abnormal' and model_label == 'abnormal':
                classification = 'True Positive'
            elif gold_label == 'normal' and model_label == 'normal':
                classification = 'True Negative'
            elif gold_label == 'normal' and model_label == 'abnormal':
                classification = 'False Positive'
            elif gold_label == 'abnormal' and model_label == 'normal':
                classification = 'False Negative'
            else:
                raise ValueError("Unexpected classification values encountered.")
            
            # Format the output line
            result_line = f"{condition}: {classification} (Gold: {gold_label}, Model: {model_label})"
            results.append(result_line)
        
        return "\n".join(results)



if __name__ == "__main__":

    # Load config file
    config = load_config(config_path = 'config.json')

    # Setup API Keys
    os.environ["OPENAI_API_KEY"] = config['setup']['OpenAI_API_key']
    os.environ["ANTHROPIC_API_KEY"] = config['setup']['Anthropic_API_key']
    os.environ["AWS_API_KEY"] = config['setup']['AWS_API_key']

    # Either load canine thorax, canine abdomen, or feline thorax radiology scans and prompt path
    if config['setup']['scan_type'] == "canine_thorax":
        conditions = config['conditions']['canine_thorax_conditions']
        prompt_path = config['prompts']['canine_thorax']
    elif config['setup']['scan_type'] == "canine_abdomen":
        conditions = config['conditions']['canine_abdomen_conditions']
        prompt_path = config['prompts']['canine_abdomen']
    else:
        conditions = config['conditions']['feline_thorax_conditions']
        prompt_path = config['prompts']['feline_thorax']

    # Initialize DataProcessor module
    myDP = DataProcessor(datapath = config['setup']['input_datapath'],
                        conditions = conditions)

    # Either process and save the data or load pre-processed data
    if config['setup']['load_or_save'] == "save":
    
        # Clean and group data
        myDP.data_to_df()
        myDP.clean_and_group_data()

        # Save data to .csv file
        myDP.cleaned_data.to_csv(config['setup']['cleaned_datapath'])
        data = myDP.cleaned_data
    else:
        data = pd.read_csv(config['setup']['cleaned_datapath'])

    # Determine whether report will be compared against AI or gold labels
    if config['setup']['gold_or_ai'] == "ai":
        gold_or_ai = "ai"
    else:
        gold_or_ai = "gold"
        gold_data = myDP.clean_gold_data(config['setup']['gold_datapath'])

    # Initialize Agents module
    myAgents = Agents(model = config['setup']['model_name'],
                      conditions = conditions)

    # Reflexion model
    if config['setup']['model_type'] == "Reflexion":

        # Load prompts
        prompt_path = prompt_path['Reflexion']

        # Process the prompts to be used by the model
        myAgents.process_Reflexion_prompts(actor_path = prompt_path['actor_path'],
                                evaluator_path = prompt_path['evaluator_path'],
                                selfreflection_path = prompt_path['selfreflector_path'])

        # Go through each report pair in the dataset
        for index, row in data.iterrows():
            rad_report = row["Full Report (original radiologist report)"]

            # Get second report as AI report or gold labels
            if gold_or_ai == "ai":

                # Assign report depending on whether Report B is a report or labels
                if config['setup']['ai_report_or_labels'] == 'report':
                    report2 = row["Full Report (AI report)"]
                else:
                    report2 = row["Conditions (AI report)"]
            
            else:
                # Get compiled gold labels from gold_data dataframe
                report2 = gold_data.loc[index, "Gold Conditions"]

            print("Working on report ID " + str(row["CaseID"]))

            # Use Reflexion technique to generate assessments on the similarities/differences of reports
            assessment = myAgents.Reflexion_model(radiologist_report = rad_report, 
                                                  gold_or_ai = gold_or_ai,
                                                  report2 = report2, 
                                                  ai_report_or_labels = config['setup']['ai_report_or_labels'], 
                                                  ID = str(row["CaseID"]))

            print(assessment)

            # Update Dataframe with decisions
            data.at[index, "Difference Assessment"] = str(assessment)

            report_max = int(config['setup']['number_of_reports'])

            # Early ending to keep API costs low
            if index == report_max - 1:
                
                data.to_csv(config['setup']['results_datapath'])
                break
    
    # Chain of Thought model
    else:

        # Process the prompts to be used by the model
        prompt_path = prompt_path['ChainofThought']
        myAgents.process_CoT_prompts(cond_path = prompt_path['condition_id_path'])

        # Go through each report pair in the dataset
        for index, row in data.iterrows():
            rad_report = row["Full Report (original radiologist report)"]

            # Get second report as AI report or gold labels
            if gold_or_ai == "ai":

                # Assign report depending on whether Report B is a report or labels
                if config['setup']['ai_report_or_labels'] == 'report':
                    report2 = row["Full Report (AI report)"]
                else:
                    report2 = row["Conditions (AI report)"]
            
            else:
                # Get compiled gold labels from gold_data dataframe
                report2 = gold_data.loc[index, "Gold Conditions"]

            print("Working on report ID " + str(row["CaseID"]))

            # Use Reflexion technique to generate assessments on the similarities/differences of reports
            assessment = myAgents.ChainofThought(radiologist_report = rad_report,
                                                 gold_or_ai = gold_or_ai, 
                                                 report2 = report2,
                                                 ai_report_or_labels = config['setup']['ai_report_or_labels'])

            print(assessment)

            # Update Dataframe with decisions
            data.at[index, "Difference Assessment"] = str(assessment)

            report_max = int(config['setup']['number_of_reports'])

            # Early ending to keep API costs low
            if index == report_max - 1:
                
                data.to_csv(config['setup']['results_datapath'])
                break
    
    # Create confusion matrix using resulting data
    print("Generating confusion matrix:")
    myDP.Confusion_Matrix(data = data,
                          num_rows = config['setup']['number_of_reports'],
                          condition_list = conditions,
                          matrix_path = config['setup']['matrix_path'])





                






        

        



    
        