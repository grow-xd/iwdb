import Image from 'next/image';
import Link from 'next/link';
import React from 'react';

export default function WebLinkCard({website}) {
    return (
        <Link href={`/details/${website.id}`}>
            <div className="dark:shadow-dark-xl h-64 border border-neutral-300 dark:border-neutral-600 rounded-[13px] p-2 hover:shadow-xl transition duration-300 ease-in-out cursor-pointer">
                <div className="w-full h-[65%] bg-neutral-200 overflow-hidden rounded-[10px]">
                    <Image
                        src={website.image || '/default.jpeg'}
                        alt={website.title || 'Website Image'}
                        width={1440}
                        height={1440}
                        className="h-full w-full object-cover object-top"
                    />
                </div>
                <div className="p-1 flex flex-col gap-1 mt-1">
                    <h2 className="text-sm truncate">{website.title}</h2>
                    <h3 className="text-sm truncate">{website.url}</h3>
                    <div className="flex items-center justify-between gap-3">
                        <div className="flex gap-1 items-center mt-1">
                            <h3 className="text-sm">Rating: </h3>
                            {
                                Array.from({ length: website.rating }).map((_, i) => (
                                    <div key={i} className="h-4 w-4">
                                        <Image src={'/star.png'} alt="alt" width={1440} height={1440} />
                                    </div>
                                ))
                            }
                        </div>
                        <div className={`h-4 w-4 rounded-[5px] ${website.trust_score >= 80 && website.trust_score <= 100 && 'bg-green-500'} ${website.trust_score >= 50 && website.trust_score < 80 && 'bg-amber-400'} ${website.trust_score >= 0 && website.trust_score < 50 && 'bg-red-500'} `}></div>
                    </div>
                </div>
            </div>
        </Link>
    );
}
