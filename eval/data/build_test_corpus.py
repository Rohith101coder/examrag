"""Script to generate 3 standardized, open-licensed academic textbooks for evaluation."""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.pdfgen import canvas

DATA_DIR = Path(__file__).resolve().parent

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#555555"))
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 36, page_text)
        self.drawString(54, 36, "ExamRAG Benchmark Corpus | Open Educational Resource (CC BY-SA 4.0)")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 48, 612 - 54, 48)
        self.restoreState()

def build_pdf(filename: str, title: str, pages_content: list[dict]):
    target_path = DATA_DIR / filename
    doc = SimpleDocTemplate(
        str(target_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=60
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=12
    )
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=8
    )

    story = []
    for i, p in enumerate(pages_content):
        if i > 0:
            story.append(PageBreak())
        story.append(Paragraph(f"{title} — {p['header']}", title_style))
        story.append(Spacer(1, 4))
        for section in p['sections']:
            story.append(Paragraph(section['title'], heading_style))
            for para in section['paragraphs']:
                story.append(Paragraph(para, body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated {filename} ({len(pages_content)} pages).")

def main():
    # 1. CPU Scheduling (5 pages)
    sched_pages = [
        {
            "header": "Fundamentals of CPU Scheduling",
            "sections": [
                {
                    "title": "1.1 Basic Concepts & Multiprogramming",
                    "paragraphs": [
                        "CPU scheduling is the basis of multiprogramming operating systems. By switching the CPU among processes, the operating system can make the computer more productive. In a uniprocessor system, only one process can run at any time; others must wait until the CPU is free and can be rescheduled.",
                        "The success of CPU scheduling depends on the observed property of processes: process execution consists of a cycle of CPU execution and I/O wait. Processes alternate between these two states. Process execution begins with a CPU burst, followed by an I/O burst, followed by another CPU burst, eventually ending on a CPU burst with a system request to terminate."
                    ]
                },
                {
                    "title": "1.2 Scheduling Criteria",
                    "paragraphs": [
                        "Different CPU scheduling algorithms have different properties, and the choice of an algorithm depends on various criteria. CPU utilization is the percentage of time the CPU is busy, with typical real-world values ranging from 40 percent in lightly loaded systems to 90 percent in heavily loaded environments.",
                        "Throughput represents the number of processes completed per unit of time. Turnaround time is the interval from the time of submission of a process to the time of its completion. Waiting time is the sum of periods spent waiting in the ready queue. Response time is the time from the submission of a request until the first response is produced."
                    ]
                }
            ]
        },
        {
            "header": "FCFS and SJF Scheduling",
            "sections": [
                {
                    "title": "2.1 First-Come, First-Served (FCFS) Scheduling",
                    "paragraphs": [
                        "The simplest CPU scheduling algorithm is the First-Come, First-Served (FCFS) algorithm. With this scheme, the process that requests the CPU first is allocated the CPU first. The implementation of the FCFS policy is easily managed with a FIFO queue. When a process enters the ready queue, its PCB is linked onto the tail of the queue.",
                        "The average waiting time under the FCFS policy is generally not minimal and may vary substantially if process CPU burst times vary greatly. In particular, the convoy effect occurs when all other processes wait for one big process to get off the CPU, resulting in lower CPU and device utilization than might be possible if shorter processes were allowed to proceed first."
                    ]
                },
                {
                    "title": "2.2 Shortest-Job-First (SJF) and SRTF",
                    "paragraphs": [
                        "The Shortest-Job-First (SJF) algorithm associates with each process the length of its next CPU burst. When the CPU is available, it is assigned to the process that has the smallest next CPU burst. If the next CPU bursts of two processes are the same, FCFS scheduling is used to break the tie.",
                        "SJF is provably optimal because it yields the minimum average waiting time for a given set of processes. The preemptive version of SJF is known as Shortest-Remaining-Time-First (SRTF). In SRTF, if a newly arrived process has a shorter remaining burst time than the currently executing process, the CPU preempts the current process."
                    ]
                }
            ]
        },
        {
            "header": "Round Robin and Priority Scheduling",
            "sections": [
                {
                    "title": "3.1 Round Robin (RR) Algorithm",
                    "paragraphs": [
                        "The Round Robin (RR) scheduling algorithm is designed specifically for time-sharing systems. It is similar to FCFS scheduling, but preemption is added to switch between processes. A small unit of time, called a time quantum or time slice, is defined, generally ranging from 10 to 100 milliseconds.",
                        "The ready queue is treated as a circular FIFO queue. The CPU scheduler goes around the ready queue, allocating the CPU to each process for a time interval of up to 1 time quantum. If the process has a CPU burst greater than 1 time quantum, the process is preempted and put back at the tail of the ready queue. The performance of RR depends heavily on the size of the time quantum: if the quantum is extremely large, RR behaves like FCFS; if it is extremely small, context-switch overhead degrades system throughput."
                    ]
                },
                {
                    "title": "3.2 Priority Scheduling & Starvation",
                    "paragraphs": [
                        "In Priority scheduling, an integer priority is associated with each process, and the CPU is allocated to the process with the highest priority (where lower numbers typically denote higher priority). Equal-priority processes are scheduled in FCFS order.",
                        "A major problem with priority scheduling algorithms is indefinite blocking, or starvation. A process that is ready to run can linger indefinitely if higher-priority processes keep arriving. A well-known solution to starvation is aging, a technique of gradually increasing the priority of processes that wait in the system for a long time."
                    ]
                }
            ]
        },
        {
            "header": "Multilevel Queues & Multiprocessor Scheduling",
            "sections": [
                {
                    "title": "4.1 Multilevel Queue and Feedback Queue",
                    "paragraphs": [
                        "A Multilevel Queue (MLQ) scheduling algorithm partitions the ready queue into several separate queues based on process properties like memory size, process priority, or process type (such as interactive foreground versus batch background processes). Each queue has its own scheduling algorithm, such as RR for foreground and FCFS for background.",
                        "The Multilevel Feedback Queue (MLFQ) algorithm allows a process to move between queues. If a process uses too much CPU time, it is moved to a lower-priority queue. This scheme leaves I/O-bound and interactive processes in the higher-priority queues. In addition, a process that waits too long in a lower-priority queue may be moved to a higher-priority queue through aging, preventing starvation."
                    ]
                },
                {
                    "title": "4.2 Multiprocessor Scheduling & Affinity",
                    "paragraphs": [
                        "When multiple CPUs are available, load sharing becomes possible, but the scheduling problem becomes correspondingly more complex. In asymmetric multiprocessing, one master processor handles all scheduling decisions, I/O processing, and other system activities. In symmetric multiprocessing (SMP), each processor is self-scheduling.",
                        "Processor affinity refers to a processor's preference to keep a thread running on the same processor to avoid the high cost of invalidating and repopulating cache memory. Soft affinity occurs when an OS attempts to keep a process on a processor without guaranteeing it; hard affinity allows a process to specify a strict subset of processors on which it can run."
                    ]
                }
            ]
        },
        {
            "header": "Real-Time Scheduling and Algorithm Evaluation",
            "sections": [
                {
                    "title": "5.1 Real-Time CPU Scheduling",
                    "paragraphs": [
                        "Soft real-time systems provide no guarantee as to when a critical real-time process will be scheduled; they guarantee only that the process is given preference over noncritical processes. Hard real-time systems have strict deadlines; a task must be serviced by its deadline or the system fails.",
                        "In Rate-Monotonic (RM) scheduling, periodic tasks are assigned priorities statically according to their period: the shorter the period, the higher the priority. Earliest-Deadline-First (EDF) scheduling dynamically assigns priorities according to deadlines: the earlier the deadline, the higher the priority. Unlike RM, EDF does not require tasks to be periodic."
                    ]
                },
                {
                    "title": "5.2 Algorithmic Comparison Summary",
                    "paragraphs": [
                        "Deterministic modeling is an analytic evaluation method that takes a particular predetermined workload and defines the performance of each algorithm for that workload. Simulation involves programming a model of the computer system to measure waiting and turnaround times against synthetic distributions.",
                        "Summary of trade-offs: FCFS has zero preemption overhead but suffers from convoy effect. SJF minimizes average waiting time but requires predicting future CPU burst durations. RR provides fair response times but depends critically on quantum selection. MLFQ provides dynamic adaptation for mixed interactive and computational workloads."
                    ]
                }
            ]
        }
    ]

    # 2. Memory Management (5 pages)
    mem_pages = [
        {
            "header": "Address Binding and Memory Spaces",
            "sections": [
                {
                    "title": "1.1 Logical versus Physical Address Space",
                    "paragraphs": [
                        "An address generated by the CPU is commonly referred to as a logical address or virtual address, whereas an address seen by the memory unit—that is, the address loaded into the memory address register (MAR) of the memory—is commonly referred to as a physical address.",
                        "The compile-time and load-time address-binding methods generate identical logical and physical addresses. However, execution-time address-binding schemes result in differing logical and physical addresses. The run-time mapping from virtual to physical addresses is handled by a hardware device called the Memory Management Unit (MMU)."
                    ]
                },
                {
                    "title": "1.2 Dynamic Relocation & Relocation Register",
                    "paragraphs": [
                        "The simplest MMU scheme uses a relocation register (or base register). The value in the relocation register is added to every logical address generated by a user process at the time the address is sent to memory. The user program never sees real physical memory addresses.",
                        "Dynamic loading ensures that a routine is not loaded until it is called, which improves memory-space utilization. Dynamic linking postpones the linking of libraries until execution time, allowing libraries to be shared among multiple processes via shared stubs."
                    ]
                }
            ]
        },
        {
            "header": "Contiguous Allocation & Fragmentation",
            "sections": [
                {
                    "title": "2.1 Contiguous Allocation Strategies",
                    "paragraphs": [
                        "In contiguous memory allocation, each process is contained in a single contiguous section of memory. Memory is usually partitioned into fixed or variable-sized partitions. In variable-partition schemes, the operating system keeps a table indicating which parts of memory are available and which are occupied.",
                        "When allocating a block from available free holes, three primary dynamic storage-allocation algorithms are used: First-Fit allocates the first hole that is big enough; Best-Fit allocates the smallest hole that is big enough; Worst-Fit allocates the largest available hole. First-Fit and Best-Fit are generally faster and produce better memory utilization than Worst-Fit."
                    ]
                },
                {
                    "title": "2.2 Internal and External Fragmentation",
                    "paragraphs": [
                        "External fragmentation exists when there is enough total memory space to satisfy a request, but the available spaces are not contiguous: storage is fragmented into a large number of small holes. The 50-percent rule states that for First-Fit, given N allocated blocks, 0.5N blocks will be lost to fragmentation.",
                        "Internal fragmentation occurs when memory is allocated in fixed-size blocks and the process requires less memory than the block size; the unused portion within the allocated block is wasted. Compaction is a solution to external fragmentation where memory contents are rearranged to place all free memory together in one large block, but it is possible only if relocation is dynamic and done at execution time."
                    ]
                }
            ]
        },
        {
            "header": "Paging Architecture & Translation",
            "sections": [
                {
                    "title": "3.1 Paging Mechanism",
                    "paragraphs": [
                        "Paging is a memory-management scheme that permits the physical address space of a process to be noncontiguous. Paging avoids external fragmentation and the need for compaction. Physical memory is broken into fixed-sized blocks called frames, and logical memory is broken into blocks of the same size called pages.",
                        "Every address generated by the CPU is divided into two parts: a page number (p) and a page offset (d). The page number is used as an index into a per-process page table. The page table contains the base address of each page in physical memory (frame number f), which is combined with the page offset to define the physical memory address."
                    ]
                },
                {
                    "title": "3.2 Hardware Support and Page Table Base Register",
                    "paragraphs": [
                        "The page table is kept in main memory, and a Page-Table Base Register (PTBR) points to the starting location of the page table in memory. Changing page tables requires changing only this one register, substantially reducing context-switch time.",
                        "However, the PTBR scheme has a serious disadvantage: accessing a memory location requires two memory accesses—one to fetch the page-table entry and one to fetch the actual desired data. This doubles the memory access latency unless specialized hardware caching is implemented."
                    ]
                }
            ]
        },
        {
            "header": "Translation Lookaside Buffer (TLB) & Multi-Level Paging",
            "sections": [
                {
                    "title": "4.1 Translation Lookaside Buffer (TLB)",
                    "paragraphs": [
                        "To overcome the two-memory-access penalty of page tables, computer hardware uses a special, small, fast-lookup hardware cache called a Translation Lookaside Buffer (TLB). The TLB is associative, high-speed memory composed of keys and values. When a virtual address is presented to the MMU, the page number is compared simultaneously against all TLB entries.",
                        "If the page number is found in the TLB, a TLB hit occurs, and the frame number is immediately available. If the page number is not found, a TLB miss occurs; the MMU accesses the page table in main memory, retrieves the frame number, and updates the TLB. Effective Memory Access Time (EMAT) is computed as: EMAT = (Hit_Ratio * (TLB_Search_Time + Memory_Access_Time)) + ((1 - Hit_Ratio) * (TLB_Search_Time + 2 * Memory_Access_Time))."
                    ]
                },
                {
                    "title": "4.2 Hierarchical and Inverted Page Tables",
                    "paragraphs": [
                        "Most modern computers support a large logical address space (e.g., 2^32 or 2^64 bytes). In such systems, the page table itself becomes excessively large. One solution is two-level paging (hierarchical paging), where the page table itself is paged.",
                        "An Inverted Page Table has one entry for each real page of physical memory (frame), rather than one entry for each page of logical address space. Each entry contains the virtual address of the page stored in that frame, along with information about the process that owns the page, substantially reducing memory consumption."
                    ]
                }
            ]
        },
        {
            "header": "Virtual Memory and Page Replacement",
            "sections": [
                {
                    "title": "5.1 Virtual Memory and Demand Paging",
                    "paragraphs": [
                        "Virtual memory involves the separation of logical memory as perceived by users from physical memory. This separation allows an extremely large virtual memory to be provided for programmers when only a smaller physical memory is available.",
                        "In Demand Paging, pages are loaded only when they are demanded during program execution. If a process attempts to access a page marked invalid in the page table, a page fault trap occurs. The operating system handles the trap by finding a free frame in physical memory, reading the desired page from secondary backing storage into the frame, updating the page table to valid, and restarting the instruction."
                    ]
                },
                {
                    "title": "5.2 Page Replacement Algorithms & Thrashing",
                    "paragraphs": [
                        "When no free frame is available, page replacement is required. The FIFO algorithm replaces the oldest page; it is simple but subject to Belady's Anomaly, where page fault rates increase as allocated frames increase. The Optimal (OPT) algorithm replaces the page that will not be used for the longest period of time; it is provably optimal but impossible to implement in practice because it requires future knowledge.",
                        "The Least Recently Used (LRU) algorithm associates with each page the time of that page's last use; when a page must be replaced, LRU chooses the page that has not been used for the longest period. Thrashing occurs when a process spends more time paging (swapping pages in and out) than executing instructions, causing CPU utilization to collapse."
                    ]
                }
            ]
        }
    ]

    # 3. Deadlocks & Concurrency (5 pages)
    deadlock_pages = [
        {
            "header": "Concurrency and the Critical Section Problem",
            "sections": [
                {
                    "title": "1.1 The Critical Section Problem",
                    "paragraphs": [
                        "A critical section is a segment of code in which a process accesses shared resources, such as shared variables, files, or tables. When one process is executing in its critical section, no other process is allowed to execute in its critical section. Concurrent access to shared data may result in data inconsistency, termed a race condition.",
                        "A valid solution to the critical section problem must satisfy three mandatory criteria: (1) Mutual Exclusion: if process Pi is executing in its critical section, no other processes can execute in their critical sections; (2) Progress: if no process is in its critical section and some processes wish to enter, only those processes not in their remainder sections can participate in deciding who enters next; (3) Bounded Waiting: there must exist a bound on the number of times other processes are allowed to enter their critical sections after a process has made a request to enter."
                    ]
                },
                {
                    "title": "1.2 Peterson's Software Solution",
                    "paragraphs": [
                        "Peterson's solution is a classic software-based solution to the critical section problem restricted to two processes, P0 and P1. It requires two shared data items: an integer variable 'turn' and a boolean array 'flag[2]'.",
                        "The variable 'turn' indicates whose turn it is to enter the critical section. The 'flag' array indicates if a process is ready to enter its critical section. Process Pi sets flag[i] = true and turn = j. Pi waits while flag[j] == true and turn == j. Peterson's algorithm provably guarantees mutual exclusion, progress, and bounded waiting on sequential-consistency hardware architectures."
                    ]
                }
            ]
        },
        {
            "header": "Hardware Synchronization & Semaphores",
            "sections": [
                {
                    "title": "2.1 Hardware Support for Synchronization",
                    "paragraphs": [
                        "Modern computer architectures provide special hardware instructions that allow software to test and modify the content of a word, or to swap the contents of two words, atomically (as one uninterruptible unit). Two fundamental primitives are TestAndSet and CompareAndSwap.",
                        "A Mutex (mutual exclusion) lock is the simplest synchronization tool. A process must acquire the lock before entering its critical section and release it when exiting. Mutex locks that require busy waiting are called spinlocks; they avoid context-switch overhead when lock holding times are brief."
                    ]
                },
                {
                    "title": "2.2 Semaphores: Counting & Binary",
                    "paragraphs": [
                        "A Semaphore is a synchronization tool that provides more sophisticated ways for process synchronization. A semaphore S is an integer variable accessed only through two standard atomic operations: wait() (historically termed P) and signal() (historically termed V).",
                        "In counting semaphores, the value can range over an unrestricted integer domain, controlling access to a resource with a finite number of instances. In binary semaphores, the value can range only between 0 and 1, functioning identically to a mutex lock. To avoid busy waiting in semaphores, the wait() operation suspends the calling process, placing it on a waiting list associated with the semaphore."
                    ]
                }
            ]
        },
        {
            "header": "Classic Problems of Synchronization",
            "sections": [
                {
                    "title": "3.1 The Bounded-Buffer (Producer-Consumer) Problem",
                    "paragraphs": [
                        "The Bounded-Buffer problem represents a common paradigm for cooperating processes. A producer process produces information that is consumed by a consumer process. The pool consists of n buffers, each capable of holding one item.",
                        "Three semaphores are used: a mutex semaphore initialized to 1 for mutual exclusion; an 'empty' counting semaphore initialized to n representing the count of empty buffer slots; and a 'full' counting semaphore initialized to 0 representing the count of filled buffer slots. The producer waits on empty and mutex, deposits an item, and signals mutex and full."
                    ]
                },
                {
                    "title": "3.2 Readers-Writers and Dining Philosophers",
                    "paragraphs": [
                        "In the Readers-Writers problem, a database is shared among concurrent processes. Readers only read the database, whereas writers read and write. If a writer and another process access the database simultaneously, chaos ensues. Solutions require that multiple readers can access concurrently, but writers demand exclusive mutual exclusion.",
                        "The Dining Philosophers problem illustrates the challenges of allocating several resources among several processes in a deadlock-free and starvation-free manner. Five philosophers sit around a table with five chopsticks. Each philosopher needs both the left and right chopsticks to eat. If every philosopher picks up their left chopstick simultaneously, a circular wait deadlock occurs."
                    ]
                }
            ]
        },
        {
            "header": "Deadlock System Model and Characterization",
            "sections": [
                {
                    "title": "4.1 The Deadlock Problem",
                    "paragraphs": [
                        "In a multiprogramming environment, several processes may compete for a finite number of resources. A process requests resources; if the resources are not available at that moment, the process enters a waiting state. A deadlock occurs when every process in a set of processes is waiting for an event that can be caused only by another process in the set.",
                        "Resources in a computer system are categorized into resource types (e.g., CPU cycles, memory space, I/O devices), each consisting of a number of identical instances."
                    ]
                },
                {
                    "title": "4.2 The Four Coffman Conditions & Resource Graphs",
                    "paragraphs": [
                        "A deadlock situation can arise if and only if the following four Coffman conditions hold simultaneously in a system: (1) Mutual Exclusion: at least one resource must be held in a nonshareable mode; (2) Hold and Wait: a process must be holding at least one resource and waiting to acquire additional resources held by other processes; (3) No Preemption: resources cannot be preempted; (4) Circular Wait: a closed chain of processes exists such that each process holds at least one resource that is needed by the next process in the chain.",
                        "Deadlocks can be described precisely in terms of a directed graph called a Resource-Allocation Graph (RAG). The graph consists of vertices representing processes (circles) and resource types (rectangles), and directed edges representing requests (P -> R) and assignments (R -> P). If the graph contains no cycles, the system is definitely not deadlocked. If the graph contains a cycle and each resource type has only one instance, a deadlock definitely exists."
                    ]
                }
            ]
        },
        {
            "header": "Deadlock Prevention, Avoidance, and Recovery",
            "sections": [
                {
                    "title": "5.1 Deadlock Prevention",
                    "paragraphs": [
                        "Deadlock prevention provides a set of methods for ensuring that at least one of the four necessary Coffman conditions cannot hold. By preventing mutual exclusion (e.g., using spooling for printers), hold and wait (requiring processes to request all resources at once), no preemption (preempting held resources if an additional request cannot be granted), or circular wait (imposing a global linear ordering on all resource types), deadlock is impossible.",
                        "However, deadlock prevention generally results in low device utilization and reduced system throughput due to stringent restrictions on resource requests."
                    ]
                },
                {
                    "title": "5.2 Deadlock Avoidance and Banker's Algorithm",
                    "paragraphs": [
                        "Deadlock avoidance requires that the operating system be given additional advance information concerning which resources a process will request and use during its lifetime. A state is safe if the system can allocate resources to each process in some order without deadlocking.",
                        "Dijkstra's Banker's Algorithm is a classic deadlock-avoidance algorithm for systems with multiple instances of each resource type. It maintains data structures: Available (vector of length m), Max (n x m matrix), Allocation (n x m matrix), and Need (n x m matrix where Need[i][j] = Max[i][j] - Allocation[i][j]). When a process requests resources, the algorithm simulates whether granting the request leaves the system in a safe state; if not, the process must wait. Recovery from deadlock involves process termination or resource preemption with rollback."
                    ]
                }
            ]
        }
    ]

    build_pdf("os_process_scheduling.pdf", "Operating Systems: CPU Scheduling", sched_pages)
    build_pdf("os_memory_management.pdf", "Operating Systems: Memory Management & Paging", mem_pages)
    build_pdf("os_deadlocks_concurrency.pdf", "Operating Systems: Deadlocks & Synchronization", deadlock_pages)

if __name__ == "__main__":
    main()
