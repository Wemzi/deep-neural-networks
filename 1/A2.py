import numpy as np

class StudentMissingException(Exception):
    pass




class Gradebook: 
    def __init__(self):
        self.grades = dict()


    def add_student(self,student):
            self.grades[student]=list()

    def add_grade(self,student,grade):
        if student not in self.grades:
            raise  StudentMissingException("Student not found")
        add_to_this_student = self.grades[student]
        add_to_this_student.append(grade)

    def get_all_students(self):
        returnable = list(self.grades.keys())
        returnable.sort()
        return returnable

    def get_student_grades(self,student):
        if student not in self.grades:
          raise StudentMissingException("Student not found")
        returnable = self.grades[student]
        return returnable

    def get_students_with_many_5s(self):
        c5=0
        cother=0
        out = list()
        for student in self.grades:
            for grade in self.grades[student]:
                if grade == 5:
                    c5 += 1
                else:
                    cother +=1
            if c5 > cother :
                out.append(student)
            cother = 0
            c5 = 0 

        return out

    def get_avarage_grade_per_student(self):
        summa = 0
        c = 0
        out = dict()
        for student in self.grades:
            for grade in self.grades[student]:
                c += 1
                summa += grade
            if c != 0:
                average = summa / c
            else:
                average = 0
            out[student] = average
            summa = 0
            c = 0
        return out