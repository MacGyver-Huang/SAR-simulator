//
//  linkedlist.h
//  PASSEDv4
//
//  Created by Steve Chiang on 11/15/22.
//  Copyright (c) 2022 Steve Chiang. All rights reserved.
//

#ifndef linkedlist_h
#define linkedlist_h


#include <iostream>

using namespace std;




//////////////////////////////////////////////////////////////////////////////////////////////////////////////
///                                                                                                        ///
///                                          Linked List                                                   ///
///                                                                                                        ///
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Fundmental structure Ref: http://alrightchiu.github.io/SecondRound/linked-list-xin-zeng-zi-liao-shan-chu-zi-liao-fan-zhuan.html
// Intersection & Unio  Ref: https://www.geeksforgeeks.org/union-and-intersection-of-two-linked-lists/

template<typename T>
class LinkedList;    // 為了將class LinkedList設成class ListNode的friend, 需要先宣告

template<typename T>
class ListNode{
public:
	// Default constructor
	ListNode():data(0),next(0){};
	// Constructor with input values
	ListNode(T in):data(in),next(0){};
	// Copy Constructor
	ListNode(const ListNode& in){ data=in.data; next=in.next; }
	// Get/Set data
	T& getData(){ return data; }
	// Get Next
	ListNode<T>* getNextPtr(){ return next; }
	// Hang out with LinkedList<T> class
	friend class LinkedList<T>;
private:
	T data;					// Contain data
	ListNode<T>* next;		// Point to the next node
};


template<typename T>
class LinkedList{
public:
	// Default constructor
	LinkedList():first(0),len(0){};
	// Copy constructor
	LinkedList(const LinkedList& in){
		len = in.len;
		first = in.first;
	};
	// Output stream
	friend ostream &operator<<(ostream& os, const LinkedList<T>& in){	// 印出list的所有資料
		if (in.first == 0) {						// 如果first node指向NULL, 表示list沒有資料
			os << "List is empty.\n";
		} else {
			os<<"+-------------------+"<<endl;
			os<<"|    Linked List    |"<<endl;
			os<<"+-------------------+"<<endl;
			os<<"size = "<<in.len<<endl;
			ListNode<T>* current = in.first;		// 用pointer *current在list中移動
			while (current != 0) {					// Traversal
				os << current->data;
				current = current->next;
				if(current != 0) { os << endl; }
			}
		}
		return os;
	};
	void Print() const {
		if (first == 0) {
			cout<<"List is empty."<<endl;
		}else{
			ListNode<T>* current = first;
			while (current != 0) {
				cout<<current->data;
				current = current->next;
				if(current != 0) { cout<<" > "; }
			}
			cout<<endl;
		}
	}
	// Push front of this list
	void push_front(T in){     // 在list的開頭新增node
		ListNode<T>* newNode = new ListNode<T>(in);	// 配置新的記憶體
		newNode->next = first;						// 先把first接在newNode後面
		first = newNode;							// 再把first指向newNode所指向的記憶體位置
		len++;
	};
	// Push back of this list
	void push_back(T in){      // 在list的尾巴新增node
		ListNode<T>* newNode = new ListNode<T>(in);	// 配置新的記憶體
		
		if (first == 0) {							// 若list沒有node, 令newNode為first
			first = newNode;
			len++;
			return;
		}
		
		ListNode<T>* current = first;
		while (current->next != 0) {				// Traversal
			current = current->next;
		}
		current->next = newNode;					// 將newNode接在list的尾巴
		len++;
	};
	// Get first pointer
	ListNode<T>* getFirstPtr(){ return first; }
	// Get the size of this list
	size_t size(){ return len; }
	// Delete certain index of node
	void Delete(size_t idx){	// 刪除list中第 idx 索引位置的ListNode
		if(idx > len){
			cerr<<"ERROR::Index("<<idx<<") is out of range of size = "<<len<<": Do nothing!"<<endl;
			exit(EXIT_FAILURE);
		}
		
		ListNode<T>* current = first;
		ListNode<T>* previous = 0;
		for(size_t i=0;i<idx;++i){		// Traversal
			previous = current;         // 如果current指向NULL
			current = current->next;    // 或是current->data == x
		}
		
		if (current == 0) {                 // list沒有要刪的node, 或是list為empty
			cerr<<"ERROR::List is empty!"<<endl;
			exit(EXIT_FAILURE);
		} else if (current == first) {        // 要刪除的node剛好在list的開頭
			first = current->next;          // 把first移到下一個node
			delete current;                 // 如果list只有一個node, 那麼first就會指向NULL
			current = 0;                    // 當指標被delete後, 將其指向NULL, 可以避免不必要bug
		} else {                              // 其餘情況, list中有欲刪除的node,
			previous->next = current->next; // 而且node不為first, 此時previous不為NULL
			delete current;
			current = 0;
		}
	};
	// Push back the node on the tail of index, idx
	void push_back(size_t idx, const T& in){
		if(idx > len){
			cerr<<"ERROR::Index("<<idx<<") is out of range of size = "<<len<<": Do nothing!"<<endl;
			exit(EXIT_FAILURE);
		}
		
		ListNode<T>* newNode = new ListNode<T>(in);	// 配置新的記憶體
		ListNode<T>* current = first;
		for(size_t i=0;i<idx;++i){		// Move to assigned index
			current = current->next;
		}
		ListNode<T>* next = current->next;
		current->next = newNode;					// 將newNode接在list的尾巴
		newNode->next = next;
		len++;
	};
	// Destroy all of this list
	void Clear(){               // 把整串list刪除
		while (first != 0) {            // Traversal
			ListNode<T>* current = first;
			first = first->next;
			delete current;
			current = 0;
			len = 0;
		}
	};
	// Get the node of this list
	ListNode<T>& getNode(size_t idx){
		if(idx > len){
			cerr<<"ERROR::Index("<<idx<<") is out of range of size = "<<len<<": Do nothing!"<<endl;
			exit(EXIT_FAILURE);
		}
		
		ListNode<T>* current = first;
		for(size_t i=0;i<idx;++i){		// Traversal
			current = current->next;    // 或是current->data == x
		}
		
		if (current == 0) {                 // list沒有要刪的node, 或是list為empty
			cerr<<"ERROR::List is empty!"<<endl;
			exit(EXIT_FAILURE);
		} else {                              // 其餘情況, list中有欲刪除的node,
			return *current;
		}
	}
	// Function to get intersection of two linked lists L1 and L2
	void getIntersection(LinkedList<T>& L2, LinkedList<T>& out){
		// clean all elements of out
		out.Clear();
		// vector<size_t> out;
		ListNode<T>* t1 = first;
		
		// Traverse list1 and search each element of it in
		// list2. If the element is present in list 2, then
		// insert the element to result
		while(t1 != NULL){
			if(isPresent(L2.getFirstPtr(), t1->getData())){
				out.push_back(t1->getData());
			}
			t1 = t1->getNextPtr();
		}
	}
	// Function to get union of two linked lists head1 and L2
	void getUnion(LinkedList<T>& L2, LinkedList<T>& out){
		// clean all elements of out
		out.Clear();
		// vector<size_t> out;
		ListNode<T>* t1 = first;
		ListNode<T>* t2 = L2.getFirstPtr();
		
		// Insert all elements of list1 to the result list
		while(t1 != NULL){
			out.push_back(t1->getData());
			t1 = t1->getNextPtr();
		}
		
		// Insert those elements of list2 which are not
		// present in result list
		while(t2 != NULL){
			if (!isPresent(out.getFirstPtr(), t2->getData())){
				out.push_back(t2->getData());
			}
			t2 = t2->next;
		}
	}
private:
	// A utility function that returns true if data is present in linked list else return false
	bool isPresent(ListNode<T>* L, const T& data){
		ListNode<T>* t = L;
		while(t != NULL){
			if (t->getData() == data){
				return true;
			}
			t = t->getNextPtr();
		}
		return false;
	}
private:
	size_t len;					// size是用來記錄Linked list的長度, 非必要
	ListNode<T>* first;			// list的第一個node
};



#endif
