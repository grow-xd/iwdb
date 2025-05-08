import Link from 'next/link';
import React from 'react';


export default function Menu() {
    return (
        <div className="fixed h-[7vh] w-full px-5 sm:px-10 md:px-20 lg:px-30 py-2 flex justify-center md:justify-between items-center bg-neutral-100 dark:bg-neutral-900 z-10">
            <Link href="/">
                <div className="flex gap-3 items-end select-none">
                    <div className="">
                        <h1 className="font-bold text-xl">IWDB</h1>
                    </div>
                    <div className="">
                        <h1 className="">Internet Website Database</h1>
                    </div>
                </div>
            </Link>
            <div className="hidden md:flex gap-5 text-sm">
                <Link href="/about"><div className="cursor-pointer"><h2 className="">About</h2></div></Link>
                <Link href="/guide"><div className="cursor-pointer"><h2 className="">Guide</h2></div></Link>
                <Link href="/support"><div className="cursor-pointer"><h2 className="">Support</h2></div></Link>
            </div>
        </div>
    );
}
