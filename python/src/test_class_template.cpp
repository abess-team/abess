#include <iostream>

using namespace std;

template <class T1>
class father
{
public:
    T1 name;
    void get() { cout << "this is father " << name << endl; }
    virtual void rename(T1 name) = 0;
};

class son1 : public father<int>
{
public:
    void get() { cout << "this is son." << name << endl; }
    void rename(int name) { name = name; }
};

class son2 : public father<string>
{
public:
    void get() { cout << "this is son." << name << endl; }
    void rename(string name) { name = name; }
};

template <class T1>
void f1(int a)
{
    T1 b(0);
    cout << a + b << endl;
}

int main()
{
    son1 s1;
    s1.get();

    // father<int> f1 = s1;
    // f1.get();

    father<int> *f2 = new son1();
    father<int> *f2 = new son2();
    father *f3;

    f2->get();
    int a = 10;
    f1<int>(a);
}