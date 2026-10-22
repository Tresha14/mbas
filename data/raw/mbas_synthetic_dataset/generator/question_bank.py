"""Question bank for the MBAS synthetic interview dataset.

Each question has 4 key points. For behavioural questions the 4 points are
Situation, Task, Action, Result (STAR) in that order.
"""

QUESTIONS = [
    # ---------------- HR ----------------
    dict(id="Q01", category="hr", difficulty=0.20, text="Tell me about yourself.", points=[
        "I am a final year computer engineering student with a focus on machine learning and web development.",
        "During my degree I built several projects, including a recommendation system and a college event portal.",
        "I completed an internship where I worked on backend APIs using Python and FastAPI.",
        "I enjoy solving problems and I am looking for a role where I can keep learning and contribute to real products."]),
    dict(id="Q02", category="hr", difficulty=0.30, text="Why do you want to join our company?", points=[
        "Your company works on products that millions of people use, and I want my work to have that kind of impact.",
        "I like that your teams follow modern engineering practices such as code reviews and continuous deployment.",
        "The role matches my skills in Python and data, and it offers a clear path to grow.",
        "I have read about your mentorship programme for freshers, which is important to me."]),
    dict(id="Q03", category="hr", difficulty=0.25, text="What are your strengths?", points=[
        "My main strength is that I learn new tools quickly; I picked up React in two weeks for a project.",
        "I am consistent and I break large tasks into small daily goals.",
        "I communicate clearly with teammates and keep everyone updated on progress.",
        "I stay calm under deadlines and prioritise the most important work first."]),
    dict(id="Q04", category="hr", difficulty=0.35, text="What is your biggest weakness?", points=[
        "I sometimes spend too long polishing details before sharing my work.",
        "I have started setting time limits and asking for early feedback to fix this.",
        "Earlier I hesitated to speak in large meetings, so I joined a debate club to practise.",
        "I now track my progress on these points every month."]),
    dict(id="Q05", category="hr", difficulty=0.30, text="Where do you see yourself in five years?", points=[
        "In five years I see myself as a senior engineer who owns an important part of the product.",
        "I want to deepen my expertise in machine learning systems.",
        "I would like to mentor juniors the way seniors helped me.",
        "I hope to have contributed to features that customers really value."]),
    dict(id="Q06", category="hr", difficulty=0.30, text="Why should we hire you?", points=[
        "I bring strong fundamentals in data structures, databases and system design.",
        "I have practical experience from internships and three full-stack projects.",
        "I am a quick learner and I adapt well to new teams.",
        "I am genuinely excited about this role and will put in the effort to succeed."]),
    dict(id="Q07", category="hr", difficulty=0.30, text="How do you handle pressure?", points=[
        "I handle pressure by planning my work and breaking it into small steps.",
        "During exams and a hackathon in the same week, I made a schedule and stuck to it.",
        "I focus on what I can control and communicate early if something may slip.",
        "Exercise and short breaks help me stay calm and focused."]),
    dict(id="Q08", category="hr", difficulty=0.25, text="What motivates you?", points=[
        "I am motivated by building things that people actually use.",
        "Learning something new every week keeps me excited about my work.",
        "Positive feedback from users or teammates pushes me to do better.",
        "I also enjoy the challenge of solving difficult problems."]),
    dict(id="Q09", category="hr", difficulty=0.25, text="Describe your ideal work environment.", points=[
        "My ideal environment is collaborative, where people share ideas openly.",
        "I like having clear goals but freedom in how I reach them.",
        "Regular feedback and code reviews help me improve.",
        "A culture that values learning and work-life balance suits me best."]),
    dict(id="Q10", category="hr", difficulty=0.40, text="What do you know about our company?", points=[
        "I know your company builds software products for the finance and retail sectors.",
        "You recently expanded into AI-based analytics.",
        "Your engineering blog shows a strong focus on scalability and reliability.",
        "You have offices in several Indian cities and a growing global client base."]),
    dict(id="Q11", category="hr", difficulty=0.35, text="What are your salary expectations?", points=[
        "I expect a package that is in line with industry standards for this role.",
        "I am more focused on learning and growth at this stage of my career.",
        "I am open to discussing the complete compensation including benefits.",
        "I trust the company to offer a fair package based on my performance."]),
    dict(id="Q12", category="hr", difficulty=0.20, text="Do you have any questions for us?", points=[
        "Yes, I would like to know how the team measures success in the first six months.",
        "Could you tell me about the training and mentorship for new joiners?",
        "What technologies does the team plan to adopt in the next year?",
        "How does the team handle code reviews and feedback?"]),
    # ---------------- Behavioural (STAR) ----------------
    dict(id="Q13", category="behavioural", difficulty=0.40, text="Tell me about a time you worked in a team.", points=[
        "In my third year we had to build a hospital management system as a team of four.",
        "My task was to design the database and coordinate the backend work.",
        "I set up weekly meetings, divided the modules and reviewed everyone's pull requests.",
        "We finished a week early and our project got the highest grade in the class."]),
    dict(id="Q14", category="behavioural", difficulty=0.50, text="Describe a conflict with a teammate and how you resolved it.", points=[
        "During a hackathon a teammate and I disagreed on whether to use React or plain HTML.",
        "We had only twenty four hours, so we needed to decide quickly.",
        "I suggested we list the pros and cons, and we agreed to use React for speed of development.",
        "We submitted on time and the teammate and I still work together on projects."]),
    dict(id="Q15", category="behavioural", difficulty=0.50, text="Tell me about a failure and what you learned.", points=[
        "In my second year my first machine learning project gave very poor accuracy.",
        "I had to present it to my professor within two weeks.",
        "I studied the data again, found heavy class imbalance and used resampling and better features.",
        "Accuracy improved from 55 to 82 percent and I learned to analyse data before modelling."]),
    dict(id="Q16", category="behavioural", difficulty=0.45, text="Describe a time you showed leadership.", points=[
        "I was the coordinator of our college technical fest with over a thousand participants.",
        "I was responsible for planning events and managing a team of twenty volunteers.",
        "I created a task tracker, assigned clear owners and held short daily check-ins.",
        "The fest ran smoothly and registrations increased by forty percent over the previous year."]),
    dict(id="Q17", category="behavioural", difficulty=0.45, text="Tell me about a time you met a tight deadline.", points=[
        "During my internship a client demo was moved forward by three days.",
        "I had to finish the reporting API that was still half done.",
        "I cut non-essential features, wrote tests for the core paths and worked closely with the frontend developer.",
        "The demo went well and the client approved the next phase."]),
    dict(id="Q18", category="behavioural", difficulty=0.40, text="Give an example of when you learned something quickly.", points=[
        "For a project I needed to use Docker, which I had never used before.",
        "I had one week to containerise our application.",
        "I followed the official tutorials, built small examples and asked a senior for a review.",
        "I deployed the app successfully and later wrote a guide for my classmates."]),
    dict(id="Q19", category="behavioural", difficulty=0.45, text="Describe a time you handled criticism.", points=[
        "My mentor said my code during the internship was hard to read.",
        "I needed to improve before the next sprint review.",
        "I read about clean code practices, added meaningful names and wrote smaller functions.",
        "In the next review my mentor appreciated the improvement and I now review others' code for readability."]),
    dict(id="Q20", category="behavioural", difficulty=0.45, text="Tell me about a time you went beyond expectations.", points=[
        "In a college project we only had to build a basic attendance app.",
        "I wanted it to be actually useful for teachers.",
        "I added QR check-in, monthly reports and an export to Excel after talking to two professors.",
        "The department adopted the app for one semester."]),
    dict(id="Q21", category="behavioural", difficulty=0.55, text="Describe a difficult problem you solved.", points=[
        "Our web app became very slow when the number of users increased.",
        "I had to find the cause and fix it before the final evaluation.",
        "I profiled the queries, added database indexes and introduced caching for frequent requests.",
        "Page load time dropped from six seconds to under one second."]),
    dict(id="Q22", category="behavioural", difficulty=0.50, text="Tell me about a mistake you made at work or in a project.", points=[
        "During my internship I accidentally pushed a configuration change that broke the staging server.",
        "I needed to restore it quickly and prevent it from happening again.",
        "I informed my lead immediately, rolled back the change and added a check to the deployment script.",
        "Staging was back within twenty minutes and the new check caught two similar mistakes later."]),
    dict(id="Q23", category="behavioural", difficulty=0.50, text="Tell me about a time you persuaded someone.", points=[
        "My project team wanted to skip writing tests to save time.",
        "I believed tests would actually save us time later.",
        "I showed them a small example where a test caught a bug in minutes and offered to set up the framework.",
        "The team agreed, and we had very few bugs during the final demo."]),
    dict(id="Q24", category="behavioural", difficulty=0.45, text="Describe a time you managed multiple priorities.", points=[
        "In my final semester I had exams, an internship and placement preparation at the same time.",
        "I needed to perform well in all three.",
        "I used a weekly planner, set priorities each morning and blocked time for each activity.",
        "I scored well in exams, completed the internship and cleared two placement rounds."]),
    # ---------------- Technical ----------------
    dict(id="Q25", category="technical", difficulty=0.45, text="What is the difference between a process and a thread?", points=[
        "A process is an independent program in execution with its own memory space.",
        "A thread is a lightweight unit of execution inside a process that shares the process memory.",
        "Context switching between threads is cheaper than between processes.",
        "Threads need synchronisation such as locks because they share data."]),
    dict(id="Q26", category="technical", difficulty=0.60, text="Explain normalization in databases.", points=[
        "Normalization organises tables to reduce redundancy and avoid update anomalies.",
        "First normal form requires atomic values, and second normal form removes partial dependencies on a composite key.",
        "Third normal form removes transitive dependencies between non-key attributes.",
        "Sometimes we denormalize for read performance."]),
    dict(id="Q27", category="technical", difficulty=0.55, text="What is overfitting and how do you prevent it?", points=[
        "Overfitting happens when a model learns the training data, including noise, and performs poorly on new data.",
        "We detect it when training accuracy is high but validation accuracy is low.",
        "We prevent it using more data, regularization like L2 or dropout, and early stopping.",
        "Cross-validation and simpler models also help."]),
    dict(id="Q28", category="technical", difficulty=0.45, text="Explain the time complexity of binary search.", points=[
        "Binary search works on a sorted array by repeatedly dividing the search range in half.",
        "At each step it compares the middle element with the target.",
        "Because the range halves every step, the time complexity is O(log n).",
        "The space complexity is O(1) for the iterative version."]),
    dict(id="Q29", category="technical", difficulty=0.50, text="What is the difference between TCP and UDP?", points=[
        "TCP is connection-oriented and guarantees reliable, ordered delivery of data.",
        "UDP is connectionless and does not guarantee delivery or order.",
        "TCP uses acknowledgements, retransmission and flow control, so it has more overhead.",
        "UDP is preferred for real-time applications like video calls and gaming."]),
    dict(id="Q30", category="technical", difficulty=0.45, text="Explain the four pillars of object-oriented programming.", points=[
        "The four pillars are encapsulation, abstraction, inheritance and polymorphism.",
        "Encapsulation bundles data and methods together and hides internal state.",
        "Inheritance lets a class reuse behaviour from a parent class.",
        "Polymorphism allows the same interface to have different implementations."]),
    dict(id="Q31", category="technical", difficulty=0.50, text="What is a REST API?", points=[
        "A REST API lets clients interact with resources over HTTP using standard methods.",
        "GET reads data, POST creates, PUT or PATCH updates and DELETE removes resources.",
        "REST is stateless, meaning each request contains all the information needed.",
        "Responses are usually in JSON with proper status codes."]),
    dict(id="Q32", category="technical", difficulty=0.55, text="What is the difference between SQL and NoSQL databases?", points=[
        "SQL databases store data in tables with a fixed schema and support joins and ACID transactions.",
        "NoSQL databases include document, key-value, column and graph stores with flexible schemas.",
        "NoSQL systems usually scale horizontally more easily.",
        "The choice depends on data structure, consistency needs and scale."]),
    dict(id="Q33", category="technical", difficulty=0.60, text="Explain how a hash map works.", points=[
        "A hash map stores key-value pairs and uses a hash function to compute an index for each key.",
        "Collisions are handled using chaining or open addressing.",
        "Average time for insert and lookup is O(1).",
        "When the load factor grows, the table is resized and keys are rehashed."]),
    dict(id="Q34", category="technical", difficulty=0.70, text="What is deadlock and how can it be prevented?", points=[
        "Deadlock occurs when processes wait forever for resources held by each other.",
        "The four necessary conditions are mutual exclusion, hold and wait, no preemption and circular wait.",
        "We prevent it by breaking one condition, for example by ordering resource requests.",
        "The Banker's algorithm can be used to avoid unsafe states."]),
    dict(id="Q35", category="technical", difficulty=0.60, text="Explain gradient descent.", points=[
        "Gradient descent is an optimisation algorithm that minimises a loss function.",
        "It updates parameters in the opposite direction of the gradient, scaled by a learning rate.",
        "Stochastic and mini-batch versions use subsets of data for faster updates.",
        "A learning rate that is too high can diverge, and too low makes training slow."]),
    dict(id="Q36", category="technical", difficulty=0.65, text="What happens when you type a URL in the browser and press enter?", points=[
        "The browser first resolves the domain name to an IP address using DNS.",
        "It opens a TCP connection, and for HTTPS performs a TLS handshake.",
        "It sends an HTTP request and the server returns a response with HTML.",
        "The browser parses the HTML, loads CSS and JavaScript, and renders the page."]),
]

VAGUE_SENTENCES = [
    "I think it is a very important topic.",
    "I have studied this in my college.",
    "It depends on the situation actually.",
    "I am not completely sure about the details.",
    "We used it in some of our projects.",
    "It is basically related to how things work in the system.",
    "I have read about it but I don't remember everything.",
    "There are many ways to look at it.",
    "It is something every engineer should know.",
    "I would need to check once to be exact.",
]

CONTRADICTION_SENTENCES = [
    "Actually no, I have not really worked on that myself.",
    "Well, to be honest the team did most of that part, not me.",
    "Sorry, I think what I said earlier is not fully correct.",
]

OPENERS = ["So,", "Okay,", "Yes,", "Sure,", "Right,", "Well,"]
FILLERS = ["um", "uh", "like", "basically", "actually", "you know", "matlab", "so"]
HEDGES = ["I think", "maybe", "probably", "I'm not sure but", "I guess"]
RESTART_PREFIXES = ["I was", "We did", "It is", "So the", "I mean the"]

INTERVIEWER_ACKS = [
    "Okay, thank you.", "Alright, got it.", "Thanks, that's helpful.", "Okay.",
    "Good, let's move on.", "Hmm, okay.", "Understood, thank you.",
]

MEETING_ACTION_TEMPLATES = [
    "{owner} to share the updated design document by {day}.",
    "{owner} to fix the login bug before the next standup.",
    "{owner} to follow up with the client about requirements.",
    "{owner} to prepare the demo for the review meeting.",
    "{owner} to update the test cases for the payment module.",
    "{owner} to check the database performance issue.",
    "{owner} to send meeting notes to the whole team.",
]
