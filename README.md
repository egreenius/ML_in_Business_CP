# ML_in_Business_CP
## Machine Learning in Business. Course Project: Loan Repayment Prediction

Курсовой проект по курсу "Машинное обучение в бизнесе". 

__Стек:__ python, docker

__ML:__ sklearn, pandas, numpy. API: flask. Данные: с kaggle - https://www.kaggle.com/itssuru/loan-data

__Задача:__ На основании имеющихся данных сервиса LendingClub.com требуется построить модель для прогнозирования невозврата заемщиком кредита в полном объеме.

__Используемые признаки:__

- credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
- purpose: The purpose of the loan (takes values "creditcard", "debtconsolidation", "educational", "majorpurchase", "smallbusiness", and "all_other").
- int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
- installment: The monthly installments owed by the borrower if the loan is funded.
- log.annual.inc: The natural log of the self-reported annual income of the borrower.
- dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
- fico: The FICO credit score of the borrower.
- days.with.cr.line: The number of days the borrower has had a credit line.
- revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
- revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
- inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
- delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
- pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

__Преобразования признаков:__ OHE кодирование для категориальных признаков, StandardScaler для вещественных признаков

__Модель:__ градиентный бустинг

__Инструкция по разворачиванию:__

Переходим в терминал, переходим рабочую папку, в которой хотим развернуть проект. Проверяем командой pwd, что находимся именно в этой папке. Клонируем репозиторий и создаем образ (image):

    $ git clone https://github.com/egreenius/ML_in_Business_CP.git
    $ cd ML_in_Business_CP
    $ docker build -t ml_loan_repayment_prediction .

Запускаем контейнер:

(*здесь Вам нужно проверить наличие каталога с обученной моделью на локальном компьютере (<your_local_full_path_to_pretrained_models>) и вставить полный путь к этому каталогу в следующей команде. ->*)

    $ docker run -d -p 8180:8180  -v <your_local_full_path_to_pretrained_models> :/app/app/models ml_loan_repaiment_prediction

Проверяем, запустился ли контейнер командой:

    $ docker ps

В терминале отобразится список запущенных контейнеров. Один из них должен быть контейнер с именем: ml_loan_repaiment_prediction
Используем адрес localhost:8180 для обращения к API. Для проверки, что сервер работает и отвечает на запросы, можно запутить браузер и в адресной строке указать адрес: http://localhost:8180. Если все в порядке, в браузере должен отобразиться текст:

    Welcome to prediction of loan return process. Please use 'http://
    /predict' to POST

Для осуществления прогнозов можно использовать файл step3_mlinbusiness_cp.py либо в интерактивном режиме в jupiter ноутбуке, либо в терминале, используя команду:

    $ python step3_mlinbusiness_cp.py
